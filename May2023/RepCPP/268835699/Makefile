BASE = preprocessor
CXX ?= g++-9
CXXFLAGS ?= -std=c++17
CXXFLAGS += -O1
SRCS = src/preprocessor.cc \
	src/io.cc \
	src/util.cc \
	src/clang/clang-wrapper.cc \
	src/preprocessor/cxx_ast.cc \
	src/preprocessor/graph-creator.cc \
	src/preprocessor/pragma.cc \
	src/preprocessor/types.cc \
	src/preprocessor/util.cc

all: $(BASE)

$(BASE):
	$(CXX) $(CXXFLAGS) $(SRCS) preprocessor.cpp -I/usr/lib/llvm-9/include -Iinclude -L/usr/lib/llvm-9/lib -Llib -o preprocessor -lclang -lboost_program_options -lpragma-parser

clean:
	rm -f preprocessor
