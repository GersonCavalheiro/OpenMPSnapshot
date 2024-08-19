

#ifndef GOOGLE_PROTOBUF_COMPILER_CPP_FILE_H__
#define GOOGLE_PROTOBUF_COMPILER_CPP_FILE_H__

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/compiler/cpp/cpp_field.h>
#include <google/protobuf/compiler/cpp/cpp_helpers.h>
#include <google/protobuf/compiler/cpp/cpp_options.h>

namespace google {
namespace protobuf {
class FileDescriptor;        
namespace io {
class Printer;             
}
}

namespace protobuf {
namespace compiler {
namespace cpp {

class EnumGenerator;           
class MessageGenerator;        
class ServiceGenerator;        
class ExtensionGenerator;      

class FileGenerator {
public:
FileGenerator(const FileDescriptor* file, const Options& options);
~FileGenerator();

void GenerateHeader(io::Printer* printer);

void GenerateProtoHeader(io::Printer* printer,
const string& info_path);
void GeneratePBHeader(io::Printer* printer,
const string& info_path);
void GenerateSource(io::Printer* printer);

int NumMessages() const { return message_generators_.size(); }
void GenerateSourceForMessage(int idx, io::Printer* printer);
void GenerateGlobalSource(io::Printer* printer);

private:
class ForwardDeclarations;

void GenerateSourceIncludes(io::Printer* printer);
void GenerateSourceDefaultInstance(int idx, io::Printer* printer);

void GenerateInitForSCC(const SCC* scc, io::Printer* printer);
void GenerateTables(io::Printer* printer);
void GenerateReflectionInitializationCode(io::Printer* printer);

void GenerateForwardDeclarations(io::Printer* printer);

void FillForwardDeclarations(ForwardDeclarations* decls);

void GenerateTopHeaderGuard(io::Printer* printer,
const string& filename_identifier);
void GenerateBottomHeaderGuard(io::Printer* printer,
const string& filename_identifier);

void GenerateLibraryIncludes(io::Printer* printer);
void GenerateDependencyIncludes(io::Printer* printer);

void GenerateMetadataPragma(io::Printer* printer, const string& info_path);

void GenerateGlobalStateFunctionDeclarations(io::Printer* printer);

void GenerateMessageDefinitions(io::Printer* printer);

void GenerateEnumDefinitions(io::Printer* printer);

void GenerateServiceDefinitions(io::Printer* printer);

void GenerateExtensionIdentifiers(io::Printer* printer);

void GenerateInlineFunctionDefinitions(io::Printer* printer);

void GenerateProto2NamespaceEnumSpecializations(io::Printer* printer);

void GenerateMacroUndefs(io::Printer* printer);

bool IsSCCRepresentative(const Descriptor* d) {
return GetSCCRepresentative(d) == d;
}
const Descriptor* GetSCCRepresentative(const Descriptor* d) {
return GetSCC(d)->GetRepresentative();
}
const SCC* GetSCC(const Descriptor* d) {
return scc_analyzer_.GetSCC(d);
}


const FileDescriptor* file_;
const Options options_;

SCCAnalyzer scc_analyzer_;


std::vector<MessageGenerator*> message_generators_;
std::vector<EnumGenerator*> enum_generators_;
std::vector<ServiceGenerator*> service_generators_;
std::vector<ExtensionGenerator*> extension_generators_;

std::unique_ptr<std::unique_ptr<EnumGenerator> []> enum_generators_owner_;
std::unique_ptr<std::unique_ptr<ServiceGenerator> []>
service_generators_owner_;
std::unique_ptr<std::unique_ptr<ExtensionGenerator> []>
extension_generators_owner_;

std::vector<string> package_parts_;

GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(FileGenerator);
};

}  
}  
}  

}  
#endif  
