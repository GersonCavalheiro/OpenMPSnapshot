# Default build:
# make

TARGET := drsyl
DEFINES := -DNDEBUG #-DINTSCALING

# Select compiler and linker: intel, gnu, llvm
COMPILER := llvm

# Set global defines
DEFINES += -DALIGNMENT=64

# Define dynamically linked libraries
LIBS := -lrt -lm

# ------------------------------------------------------------------------------
# Selection of flags
# ------------------------------------------------------------------------------

# Compiler-specific optimisation, reporting and linker flags
ifeq ($(COMPILER), intel)
	CC       := icc
	DEFINES  +=
	CFLAGS   := -Wall -std=gnu99 -O2 -xHost -malign-double -qopt-prefetch -qopenmp -ipo
	LDFLAGS  := -ipo -qopt-report=1 -qopt-report-phase=vec -qopenmp
	LIBS     += -mkl
else ifeq ($(COMPILER), llvm)
	CC       := clang
	DEFINES  +=
	WFLAGS   := -Wall -Werror -Wno-missing-braces -Wno-error=unknown-pragmas
	CFLAGS   := $(WFLAGS) -std=gnu99 -O3 -march=native -fopenmp -funroll-loops
	LDFLAGS  := -fopenmp -flto
	LIBS     += -lopenblas -fopenmp
else # gnu
	CC       := gcc
	DEFINES  +=
	WFLAGS   := -Wall -Werror=implicit-function-declaration -Werror=incompatible-pointer-types #-fopt-info
	CFLAGS   := $(WFLAGS) -std=gnu99 -O3 -march=native -funroll-loops -fprefetch-loop-arrays -malign-double -LNO:prefetch -pipe -fopenmp
	LDFLAGS  := -flto -O3 -fopenmp
	LIBS     += -lopenblas -fopenmp
endif


# ------------------------------------------------------------------------------
# Makefile rules and targets
# ------------------------------------------------------------------------------

# Select all C source files
SRCS := $(wildcard *.c)
OBJS := $(SRCS:.c=.o)

.SUFFIXES: .c .o

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $(TARGET) $(LIBS)

%.o : %.c
	$(CC) $(CFLAGS) $(DEFINES) -c $< -o $@

clean: 
	rm -f $(TARGET) *.o ipo_out.optrpt


.PHONY: clean
