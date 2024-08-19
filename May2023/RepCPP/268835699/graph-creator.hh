#ifndef __PREPROCESSOR_GRAPH_CREATOR_HH__
#define __PREPROCESSOR_GRAPH_CREATOR_HH__

#include <pragma-parser.hh>
#include "../clang/clang-wrapper.hh"
#include "../util.hh"
#include "util.hh"
#include "types.hh"

namespace __preprocessor__ {
namespace __graph__ {
static const std::string vertex_type_name="ompDAGGraphVertex_t";
static const std::string graph_name="ompDAGGraph";
static const std::string task_name="ompDAGTask";
static const std::string graph_type_name="ompDAGGraph_t";
static const std::string graph_type=graph_type_name+"<"+vertex_type_name+">";
static const std::string index_name="ompDAGidx_";
static const std::string graph_creation_function_name="ompDAGCreateGraph";

std::vector<std::string> block_iteration(const std::vector<std::string>& src,const for_statement_t& statement);

template <typename...Args> std::vector<std::string> vertex_type(const std::string& base_identation,const Args&... args) {
std::vector<std::string> r({base_identation+"struct "+vertex_type_name+" {"});
r.insert(r.end(),{"\t"+base_identation+args...});
r.push_back(base_identation+"};");
return r;
}
}
}
#endif
