#IDIR = headers
CC = clang++
CXXFLAGS = -std=c++20 -O3 -march=native
DEBUG_CFLAGS = -Wall -Werror -pedantic -ggdb3 -Wno-error=unknown-pragmas
SOURCE_DIR = src
HEADER_DIR = $(SOURCE_DIR)/headers

#SRC1 = $(SOURCE_DIR)/FCLayer.cpp $(SOURCE_DIR)/Output.cpp
SRC2 = main.cpp
#SRC = $(SRC1) $(SRC2)
main: $(SRC2)

.PHONY: clean
clean:
	rm -f *.o main