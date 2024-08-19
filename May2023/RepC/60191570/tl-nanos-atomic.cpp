#include "tl-pragmasupport.hpp"
#include "cxx-utils.h"
namespace TL { namespace Nanos {
#if 0
static AST_t inefficient_atomic(PragmaCustomConstruct atomic_construct)
{
fatal_error("%s: error '#pragma atomic' cannot be currently implemented for '%s'\n",
atomic_construct.get_ast().get_locus().c_str(),
atomic_construct.get_statement().prettyprint().c_str());
}
static bool allowed_expressions_critical(Expression expr, bool &using_builtin)
{
Expression::OperationKind op_kind = expr.get_operation_kind();
if (op_kind == Expression::PREINCREMENT  
|| op_kind == Expression::POSTINCREMENT 
|| op_kind == Expression::PREDECREMENT 
|| op_kind == Expression::POSTDECREMENT) 
{
Expression operand = expr.get_unary_operand();
bool is_lvalue = false;
Type t = operand.get_type(is_lvalue);
CXX_LANGUAGE()
{
if (t.is_reference())
t = t.references_to();
}
if (!is_lvalue
|| !(t.is_integral_type()
|| t.is_floating_type()))
return false;
using_builtin = t.is_integral_type();
return true;
}
if (expr.is_operation_assignment()
&& (op_kind == Expression::ADDITION 
|| op_kind == Expression::SUBSTRACTION  
|| op_kind == Expression::MULTIPLICATION  
|| op_kind == Expression::DIVISION 
|| op_kind == Expression::BITWISE_AND  
|| op_kind == Expression::BITWISE_OR  
|| op_kind == Expression::BITWISE_XOR 
|| op_kind == Expression::SHIFT_LEFT 
|| op_kind == Expression::SHIFT_RIGHT 
)
)
{
Expression lhs = expr.get_first_operand();
Expression rhs = expr.get_second_operand();
bool is_lvalue = false;
Type t = lhs.get_type(is_lvalue);
CXX_LANGUAGE()
{
if (t.is_reference())
t = t.references_to();
}
bool lhs_is_integral = t.is_integral_type();
if (!is_lvalue
|| !(lhs_is_integral
|| t.is_floating_type()))
return false;
t = rhs.get_type(is_lvalue);
if (!(t.is_integral_type()
|| t.is_floating_type()))
return false;
using_builtin =
lhs_is_integral 
&& (op_kind == Expression::ADDITION 
|| op_kind == Expression::SUBSTRACTION  
|| op_kind == Expression::BITWISE_AND  
|| op_kind == Expression::BITWISE_OR  
|| op_kind == Expression::BITWISE_XOR 
);
return true;
}
return false;
}
static AST_t compare_and_exchange(PragmaCustomConstruct atomic_construct, Expression expr)
{
Expression::OperationKind op_kind = expr.get_operation_kind();
Source critical_source;
Source type, lhs, rhs, op, bytes, proper_int_type, temporary;
Type expr_type = expr.get_type();
if (expr_type.is_reference())
{
expr_type = expr_type.references_to();
}
critical_source 
<< "{"
<<   type << " __oldval;"
<<   type << " __newval;"
<<   temporary
<<   "do {"
<<      "__oldval = (" << lhs << ");"
<<      "__newval = __oldval " << op << " (" << rhs << ");"
<<      "__sync_synchronize();"
<<   "} while (!__sync_bool_compare_and_swap_" << bytes << "( &(" << lhs << ") ,"
<<                 "*(" << proper_int_type << "*)&__oldval,"
<<                 "*(" << proper_int_type << "*)&__newval ));"
<< "}"
;
if (op_kind == Expression::PREINCREMENT  
|| op_kind == Expression::POSTINCREMENT 
|| op_kind == Expression::PREDECREMENT 
|| op_kind == Expression::POSTDECREMENT) 
{
lhs << expr.get_unary_operand().prettyprint();
rhs << "1";
if (op_kind == Expression::PREDECREMENT
|| op_kind == Expression::POSTDECREMENT)
{
op << "-";
}
else
{
op << "+";
}
}
else
{
bool is_lvalue = false;
expr.get_second_operand().get_type(is_lvalue);
lhs << expr.get_first_operand().prettyprint();
op << expr.get_operator_str();
temporary
<< type << " __temp = " << expr.get_second_operand().prettyprint() << ";";
rhs << "__temp";
}
type = expr_type.get_declaration(expr.get_scope(), "");
bytes << expr_type.get_size();
if (expr_type.get_size() == 4)
{
Type int_type(::get_unsigned_int_type());
if (int_type.get_size() == 4)
{
proper_int_type << "unsigned int";
}
else
{
internal_error("Code unreachable", 0);
}
}
else if (expr_type.get_size() == 8)
{
Type long_type(::get_unsigned_long_int_type());
Type long_long_type(::get_unsigned_long_long_int_type());
if (long_type.get_size() == 8)
{
proper_int_type << "unsigned long";
}
else if (long_long_type.get_size() == 8)
{
proper_int_type << "unsigned long long";
}
else
{
internal_error("Code unreachable", 0);
}
}
return critical_source.parse_statement(atomic_construct.get_ast(),
atomic_construct.get_scope_link());
}
static AST_t builtin_atomic_int_op(PragmaCustomConstruct atomic_construct, Expression expr)
{
Expression::OperationKind op_kind = expr.get_operation_kind();
Source critical_source;
if (op_kind == Expression::PREINCREMENT  
|| op_kind == Expression::POSTINCREMENT 
|| op_kind == Expression::PREDECREMENT 
|| op_kind == Expression::POSTDECREMENT) 
{
std::string intrinsic_function_name;
switch ((int)op_kind)
{
case Expression::PREINCREMENT:
case Expression::POSTINCREMENT:
{
intrinsic_function_name = "__sync_add_and_fetch";
break;
}
case Expression::PREDECREMENT:
case Expression::POSTDECREMENT:
{
intrinsic_function_name = "__sync_sub_and_fetch";
break;
}
default:
internal_error("Code unreachable", 0);
}
critical_source << intrinsic_function_name << "(&(" << expr.get_unary_operand() << "), 1);"
;
}
else
{
std::string intrinsic_function_name;
switch ((int)op_kind)
{
case Expression::ADDITION : 
{
intrinsic_function_name = "__sync_add_and_fetch";
break;
}
case Expression::SUBSTRACTION : 
{
intrinsic_function_name = "__sync_sub_and_fetch";
break;
}
case Expression::BITWISE_AND : 
{
intrinsic_function_name = "__sync_sub_and_fetch";
break;
}
case Expression::BITWISE_OR : 
{
intrinsic_function_name = "__sync_sub_or_fetch";
break;
}
case Expression::BITWISE_XOR : 
{
intrinsic_function_name = "__sync_sub_xor_fetch";
break;
}
default:
internal_error("Code unreachable", 0);
}
critical_source
<< "{"
<< expr.get_second_operand().get_type().get_declaration(expr.get_scope(), "__tmp") 
<< "=" << expr.get_second_operand().prettyprint() << ";"
<< intrinsic_function_name 
<< "(&(" << expr.get_first_operand() << "), __tmp);"
<< "}"
;
}
return critical_source.parse_statement(atomic_construct.get_ast(),
atomic_construct.get_scope_link());
}
void common_atomic_postorder(PragmaCustomConstruct atomic_construct)
{
Statement critical_body = atomic_construct.get_statement();
AST_t atomic_tree;
if (!critical_body.is_expression())
{
std::cerr << atomic_construct.get_ast().get_locus_str() << ": warning: 'atomic' construct requires an expression statement" << std::endl;
atomic_tree = inefficient_atomic(atomic_construct);
}
else
{
Expression expr = critical_body.get_expression();
bool using_builtin = false;
if (!allowed_expressions_critical(expr, using_builtin))
{
std::cerr << atomic_construct.get_ast().get_locus_str() << ": warning: 'atomic' expression cannot be implemented efficiently" << std::endl;
atomic_tree = inefficient_atomic(atomic_construct);
}
else
{
if (using_builtin)
{
atomic_tree = builtin_atomic_int_op(atomic_construct, expr);
std::cerr << atomic_construct.get_ast().get_locus_str() << ": info: 'atomic' construct implemented using atomic builtins" << std::endl;
}
else
{
atomic_tree = compare_and_exchange(atomic_construct, expr);
std::cerr << atomic_construct.get_ast().get_locus_str() << ": info: 'atomic' construct implemented using compare and exchange" << std::endl;
}
}
}
atomic_construct.get_ast().replace(atomic_tree);
}
#endif
} }
