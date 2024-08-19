#ifndef __PREPROCESSOR_CXX_AST_HH__
#define __PREPROCESSOR_CXX_AST_HH__

#include <pragma-parser.hh>
#include "../clang/clang-wrapper.hh"
#include "../util.hh"
#include "util.hh"
#include "types.hh"
#include "pragma.hh"

namespace __preprocessor__ {
namespace __cxx__ {
enum class variable_list_mode_e {
NAMES,
NAMES_TYPES
};
using variable_list_mode_t=variable_list_mode_e;

std::vector<for_statement_t> for_statements(const clang_ast_t& ast,clang_ast_node_t* root=nullptr);
std::vector<for_statement_t> for_statements(const std::vector<std::string>& src,const clang_ast_t& ast,clang_ast_node_t* root=nullptr);
void remove_unexposed(clang_ast_t &ast,clang_ast_node_t *root=nullptr);
variable_t variable_type(const clang_ast_node_t& node);
template <variable_list_mode_t mode=variable_list_mode_t::NAMES_TYPES,std::enable_if_t<mode==variable_list_mode_t::NAMES_TYPES,int> = 0>
std::set<std::pair<std::string,variable_t>> variable_list(const clang_ast_t& ast,clang_ast_node_t* root=nullptr) {
std::set<std::pair<std::string,variable_t>> variables;
auto visitor=[&variables](const clang_ast_node_t& node,size_t depth) {
variable_t type=variable_type(node);
if(type!=NOT_VAR)
variables.insert(std::make_pair(std::get<std::string>(node.value),type));
};
ast.traverse(visitor,root);
return variables;
}
template <variable_list_mode_t mode=variable_list_mode_t::NAMES_TYPES,std::enable_if_t<mode==variable_list_mode_t::NAMES,int> = 0>
std::set<std::string> variable_list(const clang_ast_t& ast,clang_ast_node_t* root=nullptr) {
std::set<std::string> variables;
auto visitor=[&variables](const clang_ast_node_t& node,size_t depth) {
variable_t type=variable_type(node);
if(type!=NOT_VAR)
variables.insert(std::get<std::string>(node.value));
};
ast.traverse(visitor,root);
return variables;
}
std::pair<std::map<clang_ast_node_t*,size_t>,std::vector<clang_ast_node_t*>> pragma_code_blocks(const std::vector<pragma_ast_node_t*>& pragma_translation,const clang_ast_t& ast,clang_ast_node_t* root=nullptr);
std::vector<std::string> pragma_identifier_list(const clang_ast_t& ast,clang_ast_node_t* root=nullptr);
std::string remove_tasks(const std::vector<std::string>& src,const std::vector<pragma_ast_node_t*>& pragma_translation,const std::vector<clang_ast_node_t*>& cxx_translation,const clang_ast_t& ast,clang_ast_node_t* root=nullptr,bool leave=false);

}
}
#endif
