

#include "rawspeedconfig.h"                  
#include "adt/AlignedAllocator.h"            
#include "adt/DefaultInitAllocatorAdaptor.h" 
#include "io/Buffer.h"                       
#include "io/FileIOException.h"              
#include "io/FileReader.h"                   
#include <cstdint>                           
#include <cstdlib>                           
#include <iostream>                          
#include <memory>                            
#include <string>                            
#include <tuple>                             
#include <vector>                            

#ifdef HAVE_OPENMP
#include <omp.h> 
#endif

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* Data, size_t Size);

static int usage() {
std::cout << "This is just a placeholder.\nFor fuzzers to actually function, "
"you need to build rawspeed with clang compiler, with FUZZ "
"build type.\n";

return EXIT_SUCCESS;
}

static void process(const char* filename) noexcept {
rawspeed::FileReader reader(filename);
std::unique_ptr<std::vector<
uint8_t, rawspeed::DefaultInitAllocatorAdaptor<
uint8_t, rawspeed::AlignedAllocator<uint8_t, 16>>>>
storage;
rawspeed::Buffer buf;

try {
std::tie(storage, buf) = reader.readFile();
} catch (const rawspeed::FileIOException&) {
return;
}

LLVMFuzzerTestOneInput(buf.getData(0, buf.getSize()), buf.getSize());
}

int main(int argc, char** argv) {
if (1 == argc || (2 == argc && std::string("-help=1") == argv[1]))
return usage();

#ifdef HAVE_OPENMP
const auto corpusCount = argc - 1;
auto chunkSize = (corpusCount / (10 * omp_get_num_threads()));
if (chunkSize <= 1)
chunkSize = 1;
#pragma omp parallel for default(none) firstprivate(argc, argv, chunkSize)     \
schedule(dynamic, chunkSize)
#endif
for (int i = 1; i < argc; ++i)
process(argv[i]);

return EXIT_SUCCESS;
}
