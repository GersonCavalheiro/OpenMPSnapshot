

#ifndef GOOGLE_PROTOBUF_COMPILER_CPP_OPTIONS_H__
#define GOOGLE_PROTOBUF_COMPILER_CPP_OPTIONS_H__

#include <string>

#include <google/protobuf/stubs/common.h>
namespace google {
namespace protobuf {
namespace compiler {
class AccessInfoMap;

namespace cpp {

struct Options {
Options()
: safe_boundary_check(false),
proto_h(false),
transitive_pb_h(true),
annotate_headers(false),
enforce_lite(false),
table_driven_parsing(false),
table_driven_serialization(false),
lite_implicit_weak_fields(false),
bootstrap(false),
num_cc_files(0),
access_info_map(NULL) {}

string dllexport_decl;
bool safe_boundary_check;
bool proto_h;
bool transitive_pb_h;
bool annotate_headers;
bool enforce_lite;
bool table_driven_parsing;
bool table_driven_serialization;
bool lite_implicit_weak_fields;
bool bootstrap;
int num_cc_files;
string annotation_pragma_name;
string annotation_guard_name;
const AccessInfoMap* access_info_map;
};

}  
}  
}  


}  
#endif  
