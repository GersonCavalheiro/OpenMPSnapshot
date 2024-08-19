#include "tl-lowering-visitor.hpp"
#include "tl-devices.hpp"
namespace TL { namespace Nanox {
class RemoveCopiedStuffVisitor : public Nodecl::ExhaustiveVisitor<void>
{
private:
const ObjectList<Symbol> & _symbols_to_be_removed;
public:
RemoveCopiedStuffVisitor(const ObjectList<Symbol> & symbols)
: _symbols_to_be_removed(symbols)
{
}
void visit(const Nodecl::FunctionCode& node)
{
if (_symbols_to_be_removed.contains(node.get_symbol()))
{
Nodecl::Utils::remove_from_enclosing_list(node);
}
}
void visit(const Nodecl::ObjectInit& node)
{
if (_symbols_to_be_removed.contains(node.get_symbol()))
{
Nodecl::Utils::remove_from_enclosing_list(node);
}
}
void visit(const Nodecl::CxxDecl& node)
{
if (_symbols_to_be_removed.contains(node.get_symbol()))
{
Nodecl::Utils::remove_from_enclosing_list(node);
}
}
void visit(const Nodecl::CxxDef& node)
{
if (_symbols_to_be_removed.contains(node.get_symbol()))
{
Nodecl::Utils::remove_from_enclosing_list(node);
}
}
};
void LoweringVisitor::visit(const Nodecl::OmpSs::TargetDeclaration& construct)
{
DeviceHandler device_handler = DeviceHandler::get_device_handler();
Nodecl::List symbols = construct.get_symbols().as<Nodecl::List>();
Nodecl::List devices = construct.get_devices().as<Nodecl::List>();
TL::ObjectList<Symbol> list_of_symbols;
TL::ObjectList<Nodecl::NodeclBase> declarations;
for (Nodecl::List::iterator it = symbols.begin(); it != symbols.end(); ++it)
{
Symbol symbol = it->as<Nodecl::Symbol>().get_symbol();
if (symbol.is_function()
&& !symbol.get_function_code().is_null())
{
declarations.append(symbol.get_function_code());
}
else if (symbol.is_variable())
{
declarations.append(
Nodecl::ObjectInit::make(
symbol,
construct.get_locus()));
}
else
{
if (symbol.is_defined())
{
declarations.append(
Nodecl::CxxDef::make(
nodecl_null(),
symbol,
construct.get_locus()));
}
else
{
declarations.append(
Nodecl::CxxDecl::make(
nodecl_null(),
symbol,
construct.get_locus()));
}
}
list_of_symbols.append(symbol);
}
bool using_device_smp = false;
for (Nodecl::List::iterator it = devices.begin(); it != devices.end(); ++it)
{
Nodecl::Text device_name = (*it).as<Nodecl::Text>();
using_device_smp = using_device_smp || device_name.get_text() == "smp";
DeviceProvider* device = device_handler.get_device(device_name.get_text());
ERROR_CONDITION(device == NULL,
"%s: device '%s' has not been loaded",
construct.get_locus_str().c_str(),
device_name.get_text().c_str());
device->copy_stuff_to_device_file(declarations);
}
if (!using_device_smp)
{
RemoveCopiedStuffVisitor remove_visitor(list_of_symbols);
remove_visitor.walk(CURRENT_COMPILED_FILE->nodecl);
}
Nodecl::Utils::remove_from_enclosing_list(construct);
}
} }
