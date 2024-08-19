function m2c(varargin)
% Wrapper for converting MATLAB code into a C/C++ library using MATLAB Coder
%    and optionally compiling the library into a MEX or EXE file.
%
% Usage:
%   m2c <options> topLevelFunc [-args {...}]
%
%  topLevelFunc can be the function name or file name of the top-level
%  function to be converted into C. m2c will build a C/C++ library under
%  the codegen/lib/<topLevelFunc> directory.
%
%  -args {...}
%      Specifies the data types of the MATLAB function using the same
%      format as codegen. If given, it must appear right after <topLevelFunc>.
%      If not present, the argument specification will be extracted from
%      the M file from the first comment block started with %#codegen.
%      You can specify multiple top-level functions along with their
%      arguments. In addition, you can specify additional options in the
%      %#codegen comment block, such as -lang:c++.
%
%  The options for m2c have several groups, defined as follows:
%
% COMPILATION WORKFLOW
%     -mex
%     -mex {'relative_path_to_m_file'}
%     -mex {'mex_path_and_filename'}
%           By default, m2c only generates the C code. This option requests
%           m2c to build the mex function in addition to generating the C files.
%           By default, the mex file will be saved in the same directory
%           as that of the M file. The MEX file is named <mfilename>.<mexext>.
%           If an argument is given and it ends with '/', the generated mex
%           file will be saved to the directory. The given path can be
%           either absolute or relative to the M file, but it must exist. If
%           the given argument does not end with '/', it will be treated as
%           the base naem of the MEX file and will be appended with the
%           extension <mexext> unless the extension is already given
%     -matlab
%     -matlab {'relative_path_to_m_file'}
%     -matlab {'mex_path_and_filename'}
%           Use Octave to compile MATLAB's MEX files, assuming the mex
%           command is in the path
%     -exe
%     -exe {'relative_path_to_m_file'} or
%     -exe {'exe_path_and_filename'}
%           Generate a MATLAB script for running the exe file from within
%           MATLAB for debugging in place of the M-file (for Linux and Mac).
%           By default, the exe file will be saved in the same directory
%           as the M file, with the extension '.exe'.  If a parameter is
%           given, and it ends with '/', then the executable will be saved
%           to the specified directory; if the parameter does not end with
%           '/', it will be treated as the filename of the executable.
%     -api
%     -api {'expression1, 'expression2', ...}
%           Specifies additional API functions in addition to the top-level
%           functions and include their declarations in the header file.
%           The argument list is a cell array of MATLAB regular expressions.
%           An API function must not be inlined. This can be ensured by
%           adding coder.inline('never') in the M code. Note that the
%           top-level function is automatically an API function and need
%           not be listed. This argument can be given after each top-level,
%           to make the list specific for each M file.
% COMPILATION DEPENDENCY
%     -cktop
%           By default, m2c does not regenerate C code if it already exists.
%           Use this option to request m2c to check the dependencies of the
%           top-level M file and regenerate C code. This is much faster
%           than -ckdep, so it should be the default mode during code development.
%     -ckdep
%           Use this option to request m2c to check the dependencies of the
%           M file and regenerate C code if any dependence is newer.
%           This can be slow compared to -cktop, so it should be used only if
%           you do not want to use -force.
%     -force
%           Force the regeneration of C code and recompilation.
%     -skipcg
%           Skip code generation and only perform recompilation, even if
%           -force, -ckdep, or -cktop is specified.
%
% CODE GENERATION OPTIONS
%     -c++
%     -cxx
%     -lang:c++
%           Generate C++ instead of C code.
%     -no-inf
%     -noinf
%           Disable support of NonFinite (inf and nan). It produces faster codes.
%           This is the default.
%     -inf
%           Enable support of NonFinite (inf and nan). It produces slower codes.
%     -inline
%           Enable function inlining in MATLAB Coder. This is the default
%           when -O, -O1, -O2, or -O3 is specified. It can be overwritten
%           by explicitly specifying -no-inline.
%     -no-inline
%           Disable function inlining in MATLAB Coder. This is the default
%           when -O0 is specified.
%     -replace-type
%           M2c uses the C built-in type names by default. This option requests
%           code generation to use the numeric types defined by Coder instead
%           of using the C built-in type names. This is the default if
%           -cuda is enabled for better portability.
%     -remove-emx
%           Attempt to replace all emxArrays by plain pointers, This is
%           possible if there is no reference to the size field of emxArrays
%           in the generated C code.
%     -addpath {'dir1', 'dir2', ..., 'dirN'}
%           Add the directory to the beginning of the search path for M files
%           during code generation.
%     -outdir {'dir'}
%           Specifies the output directory of the generated C code for the
%           top-level MATLAB function. By default, if -cuda-kernel is enabled,
%           the generated CUDA code will be in codegen/lib/<funcname>-cuda;
%           otherwise, the code is in codegen/lib/<funcname>. All the path
%           is relative to the M file. You can overwrite these default
%           directory names.
%     -example
%           Generate example main file.
%     -config {'expression'}
%           Use the given coder.CodeConfig object for codegen. The argument
%           is a string contained a variable or function name that evaluates
%           to a coder.CodeConfig object in the MATLAB base workspace.
%           When this option is specified, code generation is forced.
%           If string is contained in a cell array, then the string can
%           be a MATLAB expression that evaluates to a character string.
%           This overwrites the default values used by m2c, except that the
%           headers for MPI, OpenMP etc. are appended to the given CustomSourceCode.
%
% COMPILER CUSTOMIZATION
%     -cc {'expression'}
%           Specify the C compiler. It overwrites the default complier
%           choice for both the mex file and the standalone executable.
%           The cc command can be specified after it as a character
%           string, or a MATLAB expression contained in a cell array.
%     -cflags {'flag1', 'flag2', ...}
%           Specify flags for the C compiler for building the mex file or
%           standalone executable. The argument list is a cell array of
%           character strings. E.g., -cflags {'-O3', '-DNDEBUG'}. Each
%           string can be a MATLAB expression. It is appended to the flags.
%     -ldflags {'flag1', 'flag2', ...}
%           Specify flags for the linker when building the mex file or
%           standalone executable. It is appended to the flags.
%     -mexflags {'flag1', 'flag2', ...}
%           Specify flags for the mex command. It overwrites the flags
%           assigned by -O?, -cc -cflags options. The argument list is a
%           cell array of character strings. Each string can be a MATLAB
%           expression. E.g., -mexflags {'COPTIMFLAGS=''-O3 -NDEBUG'''}.
%     -D<macro>
%           Add a definition of a macro for C preprocessor.
%     -I<dir>
%           Add search path for C header files. Note that ./include
%           is added automatically if exists.
%     -l<lib>
%           Specify linking of a particular library.
%     -L<dir>
%     -L {'dir1', 'dir2', ..., 'dirn'}
%           Add search path for libraries for linking. In the second
%           format, -rpath will also be added automatically for linking
%           with shared libraries.
%
% OPTIMIZATION
%     -O0
%           Disable function inlining and variable reuse for MATLAB Coder,
%           and pass -O0 to the C compiler to disable all optimizations,
%           including all OpenMP pragmas, and it enables memory checking.
%     -O1
%           Enable function inlining for MATLAB Coder and pass the -O1
%           compiler option to the C compiler to enable basic optimization.
%           This is the default behavior if -O? and -g is not specified.
%     -O
%     -O2
%           Enable inlining and in addition allow MATLAB Coder to reuse
%           variable names freely. Ths generates C code is less readable,
%           but it may use less memory and enable better performance. It
%           passes -O2 to the C compiler to enable nearly all supported
%           optimizations that do not involve a space-speed tradeoff.
%     -O3
%           Enable level-2 optimization, and also set DynamicMemoryAllocation
%           to thresholding. It can be useful for generating codes for
%           kernel functions that require a variable-size buffer with a
%           known constant upper bound. However, it makes the code less
%           readable. It also passes -O3 to the C compiler to enable all
%           optimizations, including loop unrolling and function inlining.
% DEBUGGING
%     -g
%           Preserve MATLAB code info in C code and also generate
%           source-level debug information when compiling C.
%     -chkmem
%           Generate code for runtime error checking of buffer overflow.
%           It also enables -g, and is automatically enabled with -valgrind.
%     -gdb
%     -gdb {'expression'}
%           Generate a script for running the standalone executable in gdb.
%           The gdb command can be specified after it in a cell array.
%     -ddd
%     -ddd {'expression'}
%           Generate a script for running the standalone executable in ddd.
%           The ddd command can be specified after it in a cell array.
%     -valgrind
%     -valgrind {'expression'}
%           Generate a script for running the standalone executable in valgrind.
%           The valgrind command can be specified after it as a character
%           string, or a MATLAB expression contained in a cell array. It
%           also activates -g and -chkmem.
%
% PROFILING AND INSTRUMENTATION
%     -time
%     -time {'expression1, 'expression2', ...} [Not yet implemented]
%           List of functions for timing. The argument list may be the name
%           of a single function or a cell array of expressions that
%           evaluate to the names of the functions to be timed. These
%           functions must not be inlined. This can be ensured by
%           adding coder.inline('never') in the M fiile. If no function is
%           given after -time, then the top-level function will be timed.
%           This argument can be repeated, and then the functions will
%           be concatenated.
%     -gprof
%     -gprof {'expression'}
%           Compile standalone code with profiling support and generate
%           a script to process results from gprof (Linux only).
%           The gprof command can be specified after it as a character
%           string, or a MATLAB expression contained in a cell array.
%     -gcov
%     -gcov {'expression'}
%           Compile standalone code with profiling support and generate
%           a script to process results from gcov (Linux and Mac).
%           The gcov command can be specified after it as a character
%           string, or a MATLAB expression contained in a cell array.
%
% SUPPORTED MATH LIBRARIES
%     -mkl (or -mkl-seq)
%     -mkl-iomp (or -mkl-omp}
%     -mkl {'expression'}
%           Enable Intel MKL and link with its BLAS and sparse BLAS libraries.
%           The root directory of MKL can be specified as an express
%           contained in a cell array, or be specified by the environment
%           variables MKLROOT. By befault, -mkl specifies linking with
%           the sequential version of MKL. Use -mkl-iomp to link with
%           multithreaded MKL library with Intel OpenMP. Note that the
%           Intel OpenMP may conflict with GCC's OpenMP library, so it should
%           not be used with -omp when compiling with a non-Intel compiler.
%     -cuda
%     -cuda {'expression'}
%           Enable CUDA Toolkit and link with cuBLAS and cuSparse. The
%           root directory of CUDA can be specified by a MATLAB expression
%           contained in a cell array, or be specified by the environment
%           variables CUDA_PATH or by adding nvcc to your path. Note that
%           if CUDA is available within MATLAB (when MATLAB Parallel Toolbox
%           is installed), it will link with the CUDA libraries distributed
%           with MATLAB.
%     -petsc
%     -petsc {'expression'}
%           Enable support for PETSc. The root directory of PETSc can be
%           specified as a MATLAB expression contained in a cell array.
%           The default path can be extracted from the environment
%           variables PETSC_DIR and PETSC_ARCH.
%     -blas
%     -blas {'-Ldir1', '-llib1', '-llib2', ...}
%           Enable BLAS and link with the CBLAS library specified in
%           the list. The argument list is a cell array of character strings
%           to be passed to the C compiler and linker. Each string can be
%           a MATLAB expression. If the cell array is empty, then the
%           MATLAB built-in BLAS library will be used. This option is
%           omitted if -mkl is specified.
%     -lapack
%     -lapack {'-Ldir1', '-llib1', '-llib2', ...}
%           Enable LAPACKE and link with the LAPACK library specified in
%           the list. The argument list is a cell array of character strings
%           to be passed to the C compiler and linker. Each string can be
%           a MATLAB expression. If the cell array is empty, then the
%           MATLAB built-in LAPACK library will be used.
%
% PARALLEL PROGRAMMING
%     -omp
%     -omp {'-Ldir1', '-llib1', '-llib2', ...}
%           Enable support for OpenMP. The argument list is a cell array of
%           character strings to be passed to the C compiler and linker.
%           Each string can be a MATLAB expression. It requires MOMP and a
%           compiler that supports OpenMP. If the C code does not have any
%           parallel section but contains kernel funcions, then it will
%           create parallel sections in the driver code.
%     -nvcc or -cuda-kernel
%     -nvcc {'expression'} or -cuda-kernel {'expression'}
%           Request to compile and link a CUDA kernel code using nvcc. The
%           root directory of CUDA can be specified by a MATLAB expression
%           contained in a cell array, or be specified by the environment
%           variables CUDA_PATH or by adding nvcc to your path.
%     -mpi
%     -mpi {'-Ldir1', '-llib1', '-llib2', ...}
%           Enable support for MPI. The argument list is a cell array of
%           character strings to be passed to the C compiler and linker.
%           Each string can be a MATLAB expression. It requires MMPI.
%           This is not needed when -petsc is specified.
%
% MISCELLANEOUS
%     -v
%           Verbose mode.
%     -q
%           Quiet mode.
%     -?
%     -h
%     -help
%           Print out help.
%
% Example usage:
%       To generate code without MATLAB function inlining and with
%       default C compiler optimization flag (-O0):
%            m2c matlabfunc
%
%       To generate code with MATLAB function inlining and with
%       default C compiler optimization of level 1, 2, or 3:
%            m2c -O1 matlabfunc
%            m2c -O matlabfunc
%            m2c -O3 matlabfunc
%
%       To generate code without MATLAB function inlining, without C
%       compiler optimization, and with MATLAB code preserved in C code:
%            m2c -g matlabfunc
%
%       To generate code with MATLAB function inlining, with C compiler
%       optimization of level 1, 2 or 3, and with MATLAB code preserved in C code:
%            m2c -O1 -g matlabfunc
%            m2c -O -g matlabfunc
%            m2c -O3 -g matlabfunc
%
% See also compile, m2mex, codegen.
m2c_opts = proc_options(varargin{:});
if isempty(m2c_opts)
help m2c; %#ok<MCHLP>
return;
end
if isempty(m2c_opts.funcName)
error('Missing function in the argument list.');
end
matlabFunc = m2c_opts.funcName{1};
% Force code generation if config is specified
if ~isempty(m2c_opts.codegenConfig)
m2c_opts.force = true;
end
%% Process arguments
% Split filename into the path and filename
[mpath, func, mfile] = get_path_of_mfile(matlabFunc);
if isempty(m2c_opts.outDir)
if m2c_opts.withNvcc
m2c_opts.outDir = {['codegen/lib/' func '-cuda']};
else
m2c_opts.outDir = {['codegen/lib/' func]};
end
end
if m2c_opts.outDir{1}(1)=='/' || m2c_opts.outDir{1}(1)=='\'
cpath = [m2c_opts.outDir{1}(1) '/'];
else
cpath = [mpath m2c_opts.outDir{1} '/'];
end
if isempty(m2c_opts.codegenArgs)
% Extract arguments from source code.
m2c_opts.codegenArgs = extract_codegen_args(mfile);
if contains(m2c_opts.codegenArgs, '-lang:c++')
m2c_opts.useCpp = true;
m2c_opts.suf = 'cpp';
end
end
if m2c_opts.withNvcc
mainCFile = [cpath  func '.cu'];
mexCFile = [cpath  func '_mex.cpp'];
elseif m2c_opts.useCpp
mainCFile = [cpath  func '.cpp'];
mexCFile = [cpath  func '_mex.cpp'];
else
mainCFile = [cpath  func '.c'];
mexCFile = [cpath  func '_mex.c'];
end
regen_c = ~m2c_opts.skipcg && (m2c_opts.force || ~exist(mainCFile, 'file') || ...
~exist(mexCFile, 'file') || ~ckSignature(m2c_opts, 'codegen', mexCFile) || ...
m2c_opts.cktop && ~isnewer(mexCFile, mfile) || ...
m2c_opts.ckdep && ~ckdep(mexCFile, mfile, true));
if regen_c
% Determine whether you have codegen.
if ~exist('codegen.p', 'file')
error('m2c:MissingCoder', 'MATLAB Coder is required to generate C code.\n');
end
if ~isempty(m2c_opts.codegenConfig)
co_cfg = evalin('base', sprintf('%s', m2c_opts.codegenConfig{1}));
if isempty(co_cfg) || ~isa(co_cfg, 'coder.CodeConfig')
error('m2c:WrongArgument', ['The argument after -config must evaluate to a coder.CodeConfig ' ...
'object. Received %s Use command coder.config(''lib'') to create the object.\n'], ...
m2c_opts.codegenConfig{1});
end
else
%% Specify codegen config options
co_cfg = coder.config('lib');
co_cfg.EnableOpenMP = m2c_opts.withOMP && m2c_opts.optimLevel > 0;
co_cfg.SaturateOnIntegerOverflow = false;
co_cfg.EnableVariableSizing = true;
co_cfg.PassStructByReference = true;
co_cfg.EnableMemcpy = true;
if ~isempty(m2c_opts.dynMem) && ~m2c_opts.remEmx
co_cfg.DynamicMemoryAllocation = m2c_opts.dynMem;
else
co_cfg.DynamicMemoryAllocation = 'AllVariableSizeArrays';
end
co_cfg.FilePartitionMethod = 'SingleFile';
co_cfg.GenCodeOnly = true;
co_cfg.GenerateReport = m2c_opts.debugInfo;
co_cfg.InitFltsAndDblsToZero = true;
co_cfg.SupportNonFinite = m2c_opts.enableInf;
if m2c_opts.useCpp
co_cfg.TargetLang = 'C++';
co_cfg.TargetLangStandard = 'C++11 (ISO)';
if isprop(co_cfg, 'DynamicMemoryAllocationInterface')
co_cfg.DynamicMemoryAllocationInterface = 'C';
end
else
co_cfg.TargetLang = 'C';
co_cfg.TargetLangStandard = 'C99 (ISO)';
end
co_cfg.MultiInstanceCode = true;
co_cfg.GenerateComments = m2c_opts.debugInfo;
if ~isempty(m2c_opts.presVars)
co_cfg.PreserveVariableNames = m2c_opts.presVars;
else
co_cfg.PreserveVariableNames = 'UserNames';
end
co_cfg.MATLABSourceComments = m2c_opts.debugInfo;
try co_cfg.RuntimeChecks = m2c_opts.chkMem;
catch; end
if ~m2c_opts.exampleMain
co_cfg.GenerateExampleMain = 'DoNotGenerate';
else
co_cfg.GenerateExampleMain = 'GenerateCodeOnly';
end
co_cfg.GenerateMakefile = false;
if m2c_opts.typeRep || m2c_opts.withCuda && ~isunix()
% To support CUDA pointers, it is recommended to use the bulit-in
% definition of uint_64 for compatability with M.S. Windows,
% because MATLAB Coder uses "unsigned long" for "uint64".
co_cfg.DataTypeReplacement = 'CoderTypeDefs';
else
co_cfg.DataTypeReplacement = 'CBuiltIn';
end
if m2c_opts.withLapack
co_cfg.CustomLAPACKCallback = 'useBuiltinLAPACK';
end
end
warning('off', 'CoderFoundation:builder:TMFIncompatibilityWarningMATLABCoder');
co_cfg.CustomSourceCode = sprintf('%s\n', co_cfg.CustomSourceCode);
%% Run command
if m2c_opts.verbose
disp('Running codegen with options:');
disp(co_cfg);
end
basecommand = 'codegen -config co_cfg ';
if m2c_opts.enableInline
opts = '-O enable:inline';
else
opts = '-O disable:inline';
end
if m2c_opts.verbose; basecommand = [basecommand ' -v']; end
basecommand = [basecommand sprintf(' %s ', ...
m2c_opts.addpath{:}, m2c_opts.m2cpath{:}, '-d', m2c_opts.outDir{1})];
command = strtrim([basecommand ' ' opts ' ' func ' ' m2c_opts.codegenArgs]);
try
olddir = pwd;
if ~isempty(mpath); cd(mpath); end
if ~m2c_opts.quiet
disp(command)
end
eval(command);
cd(olddir);
catch err
cd(olddir);
error('%s', err.message);
end
% Modify C code generated by codegen
[parmode, ~, has_emxutil] = post_codegen(func, cpath, m2c_opts);
if strncmp(parmode, 'cuda', 4) && ~m2c_opts.withNvcc
error('m2c:CudaKernel', 'Cuda device code detected but -nvcc or -cuda-kernel was not specified.');
elseif strncmp(parmode, 'omp', 3) && ~m2c_opts.withOMP
warning('m2c:OMPKernel', 'OpenMP code detected but -omp was not specified.');
end
% Write Mex file to annotate the options used in codegen
writeMexFile(func, mpath, cpath, m2c_opts, parmode);
% Write README file
writeREADME(func, cpath, m2c_opts.useCpp, has_emxutil, m2c_opts.genExe, m2c_opts.withNvcc);
if exist([cpath 'rtwtypes.h'], 'file'); delete([cpath 'rtwtypes.h']); end
if exist([cpath 'defines.txt'], 'file'); delete([cpath 'defines.txt']); end
if exist([cpath '.gitignore'], 'file'); delete([cpath '.gitignore']); end
if exist([cpath 'interface'], 'dir'); rmdir([cpath 'interface'], 's'); end
if ~has_emxutil && exist([cpath func '_emxutil.h'], 'file')
delete([cpath func '_emxutil.c*']);
delete([cpath func '_emxutil.h']);
end
end
%% Generate MATLAB scripts for mex
if isoctave && ~m2c_opts.genMatlab
mexbuild = [cpath  'oct_' func '.m'];
else
mexbuild = [cpath  'mex_' func '.m'];
end
if m2c_opts.genMex
if regen_c || m2c_opts.force || ~ckSignature(m2c_opts, 'mex', mexbuild)
writeMexScript(func, mpath, cpath, m2c_opts);
if m2c_opts.withNvcc
% Also write a script for feval
writeEvalScript(func, mpath, cpath, m2c_opts);
end
end
clear(mexbuild);
if exist(mexbuild, 'file'); run(mexbuild); end
elseif m2c_opts.verbose
fprintf('To build the MEX file, use command (without quotes): "run %s".\n', ...
mexbuild);
end
%% Generate MATLAB scripts for exe if genexe is true.
if m2c_opts.genExe
if regen_c || m2c_opts.force || ~exist([cpath  func '_exe.c'], 'file')
% Write the main function for Exe
writeExeFile(func, cpath, m2c_opts);
end
if regen_c || m2c_opts.force || ~ckSignature(m2c_opts, 'exe', [cpath  'build_' func '_exe.m'])
% Write the build script for Exe
writeExeScripts(func, mpath, cpath, m2c_opts);
end
if m2c_opts.genMatlab
build_exe(cpath, func);
if m2c_opts.verbose
fprintf(['To run the EXE file in MATLAB, ', ...
'replace calls to ' func ' by run_' func '_exe.\n']);
end
elseif ~m2c_opts.quiet
fprintf('Skipping compilation of EXE file in Octave.\n');
end
end
end
function m2c_opts = proc_options(varargin)
% Process input options from left to right.
if nargin==0
m2c_opts=[];
return;
end
% If default value is {}, use [] in place of {} and re-assign default
% value to {} at the end.
m2c_opts = struct('codegenArgs', [], ...
'funcName', '', ...
'codegenConfig', '', ...
'debugInfo', false, ...
'chkMem', false, ...
'enableInline', true, ...
'enableInf', false, ...
'presVars', '', ...
'dynMem', '', ...
'optimLevel', -1, ...
'typeRep', false, ...
'remEmx', false, ...
'm2cpath', '', ...
'addpath', '', ...
'api', '', ...
'globals', '', ...
'timing', '', ...
'cppflags', '', ...
'libs', '', ...
'cc', '', ...
'cflags', '', ...
'ldflags', '', ...
'mexflags', '', ...
'withMPI', false, ...
'mpiCC', '', ...
'mpiCXX', '', ...
'mpiInc', '', ...
'mpiLibs', '', ...
'withOMP', false, ...
'ompLibs', '', ...
'withBlas', false, ...
'blasLibs', '', ...
'withLapack', false, ...
'lapackLibs', '', ...
'withMKL', false, ...
'mklDir', '', ...
'mklInc', '', ...
'mklLibs', '', ...
'withCuda', false, ...
'cudaDir', '', ...
'cudaInc', '', ...
'cudaLibs', '', ...
'withNvcc', false, ...
'withPetsc', false, ...
'petscDir', '', ...
'petscCC', '', ...
'petscCXX', '', ...
'petscCFLAGS', '', ...
'petscCXXFLAGS', '', ...
'petscInc', '', ...
'petscLibs', '', ...
'gprof', '', ...
'gcov', '', ...
'valgrind', '', ...
'gdb', '', ...
'ddd', '', ...
'outDir', '', ...
'genMatlab', ~isoctave, ...
'genMex', false, ...
'mexFile', '', ...
'genExe', false, ...
'exeFile', '', ...
'genLib', false, ...
'genDll', false, ...
'libRoot', '', ...
'useCpp', false, ...
'verbose', false, ...
'quiet', false, ...
'suf', 'c', ...
'force', false, ...
'ckdep', false, ...
'cktop', false, ...
'skipcg', false, ...
'exampleMain', false);
names = fieldnames(m2c_opts);
for i=1:length(names)
if ischar(m2c_opts.(names{i})) && isempty(m2c_opts.(names{i}))
m2c_opts.(names{i}) = {m2c_opts.(names{i})};
end
end
if contains(m2croot, ' ')
m2croot__ = ['''' m2croot ''''];
else
m2croot__ = m2croot;
end
m2c_opts.cppflags = {['-I' m2croot__ '/include']};
m2c_opts.m2cpath={'-I ./codegen'};
func_index = 0;
i = 1;
last_index = length(varargin);
while i<=last_index
opt = varargin{i};
switch opt
case '-force'
m2c_opts.force = true;
case '-ckdep'
m2c_opts.ckdep = true;
case '-cktop'
m2c_opts.cktop = true;
case '-skipcg'
m2c_opts.skipcg = true;
case {'-addpath'}
if i<last_index && varargin{i+1}(1) == '{'
dirs = eval(varargin{i+1});
i = i + 1;
elseif i<last_index-1 && varargin{i+1}(1) ~= '-' && ~isequal(varargin{i+1}(2), '-args')
dirs = varargin(i+1);
i = i + 1;
else
error('m2c:wrong_argument', ...
'Argument %s requires a cell-array argument after it.\n', opt);
end
if ~isempty(dirs)
m2c_opts.addpath = [m2c_opts.addpath  strtrim(sprintf('-I %s ', dirs{:}))];
end
case '-example'
m2c_opts.exampleMain = true;
case '-outDir'
if i<last_index && varargin{i+1}(1) == '{'
m2c_opts.outDir = eval(varargin{i+1});
i = i + 1;
elseif i<last_index-1 && varargin{i+1}(1) ~= '-' && ~isequal(varargin{i+1}(2), '-args')
m2c_opts.outDir = varargin(i+1);
i = i + 1;
else
error('m2c:wrong_argument', ...
'Argument -outdir requires a cell-array argument after it.\n');
end
case {'-mex', '-matlab'}
m2c_opts.genMatlab = m2c_opts.genMatlab || isequal(opt, '-matlab');
m2c_opts.genMex = true;
if i<last_index && varargin{i+1}(1) == '{'
m2c_opts.mexFile = eval(varargin{i+1});
i = i + 1;
elseif i<last_index-1 && varargin{i+1}(1) ~= '-' && ~isequal(varargin{i+1}(2), '-args')
m2c_opts.mexFile = varargin(i+1);
i = i + 1;
end
case '-exe'
m2c_opts.genExe = true;
if i<last_index && varargin{i+1}(1) == '{'
m2c_opts.exeFile = eval(varargin{i+1});
i = i + 1;
elseif i<last_index-1 && varargin{i+1}(1) ~= '-' && ~isequal(varargin{i+1}(2), '-args')
m2c_opts.exeFile = varargin(i+1);
i = i + 1;
end
case '-g'
m2c_opts.debugInfo = true;
case '-chkmem'
m2c_opts.chkMem = true;
m2c_opts.debugInfo = true;
case '-O0'
m2c_opts.optimLevel = 0;
m2c_opts.enableInline = false;
m2c_opts.chkMem = true;
m2c_opts.presVars = 'All';
m2c_opts.dynMem = 'AllVariableSizeArrays';
case '-O1'
% For code generation, this is the default behavior if -O? is
% not specified.
m2c_opts.optimLevel = 1;
m2c_opts.enableInline = true;
m2c_opts.presVars = 'UserNames';
m2c_opts.dynMem = 'AllVariableSizeArrays';
case {'-O', '-O2'}
m2c_opts.optimLevel = 2;
m2c_opts.enableInline = true;
m2c_opts.presVars = 'None';
m2c_opts.dynMem = 'AllVariableSizeArrays';
case '-O3'
m2c_opts.optimLevel = 3;
m2c_opts.enableInline = true;
m2c_opts.presVars = 'None';
m2c_opts.dynMem = 'Threshold';
case '-inline'
m2c_opts.enableInline = true;
case '-no-inline'
m2c_opts.enableInline = false;
case {'-noinf', '-no-inf'}
m2c_opts.enableInf = false;
case '-inf'
m2c_opts.enableInf = true;
case '-replace-type'
m2c_opts.typeRep= true;
case '-remove-emx'
m2c_opts.remEmx= true;
case '-time'
if i<last_index && varargin{i+1}(1) == '{'
m2c_opts.timing = eval(varargin{i+1});
i = i + 1;
elseif i<last_index-1 && varargin{i+1}(1) ~= '-' && ~isequal(varargin{i+1}(2), '-args')
m2c_opts.timing = varargin(i+1);
i = i + 1;
else
m2c_opts.timing = {' '};
end
case '-api'
if i<last_index && varargin{i+1}(1) == '{'
m2c_opts.api = eval(varargin{i+1});
i = i + 1;
elseif i<last_index-1 && varargin{i+1}(1) ~= '-' && ~isequal(varargin{i+1}(2), '-args')
m2c_opts.api = varargin(i+1);
i = i + 1;
else
m2c_opts.api = {' '};
end
case {'-valgrind', '-ddd', '-gdb', '-gprof', '-gcov'}
if i<last_index && varargin{i+1}(1) == '{'
m2c_opts.(opt(2:end)) = eval(varargin{i+1});
i = i + 1;
elseif i<last_index-1 && varargin{i+1}(1) ~= '-' && ~isequal(varargin{i+1}(2), '-args')
m2c_opts.(opt(2:end)) = varargin(i+1);
i = i + 1;
else
m2c_opts.(opt(2:end)) = {' '};
end
case {'-lib', '-dll'}
if i<last_index && varargin{i+1}(1) == '{'
m2c_opts.libRoot = eval(varargin{i+1});
i = i + 1;
elseif i<last_index-1 && varargin{i+1}(1) ~= '-' && ~isequal(varargin{i+1}(2), '-args')
m2c_opts.libRoot = varargin(i+1);
i = i + 1;
end
% TODO
warning('Support for ''-lib'' and ''-dll'' is not yet implemented.');
case {'-cc', '-mexflags', '-cflags', '-ldflags'}
if i<last_index && varargin{i+1}(1) == '{'
m2c_opts.(opt(2:end)) = eval(varargin{i+1});
i = i + 1;
elseif i<last_index-1 && varargin{i+1}(1) ~= '-' && ~isequal(varargin{i+1}(2), '-args')
m2c_opts.(opt(2:end)) = varargin(i+1);
i = i + 1;
else
error('m2c:wrong_argument', ...
'Argument %s requires a cell-array argument after it.\n', opt);
end
case '-L'
if i<last_index && varargin{i+1}(1) == '{'
dirs = eval(varargin{i+1});
for k=1:length(dirs)
m2c_opts.libs = [m2c_opts.libs, dylibdir(dirs{k})];
end
i = i + 1;
elseif i<last_index-1 && varargin{i+1}(1) ~= '-' && ~isequal(varargin{i+1}(2), '-args')
m2c_opts.libs = [m2c_opts.libs, dylibdir(varargin{i+1})];
i = i + 1;
else
error('m2c:wrong_argument', ...
'Argument -L requires a cell-array argument after it.\n');
end
case {'-cuda', '-nvcc', '-cuda-kernel'}
if i<last_index && varargin{i+1}(1) == '{'
m2c_opts.cudaDir = eval(varargin{i+1});
i = i + 1;
elseif i<last_index-1 && varargin{i+1}(1) ~= '-' && ~isequal(varargin{i+1}(2), '-args')
m2c_opts.cudaDir = varargin(i+1);
i = i + 1;
else
if ~isempty(getenv('CUDA_PATH'))
m2c_opts.cudaDir = {getenv('CUDA_PATH')};
elseif ismac && exist('/Developer/NVIDIA/CUDA-7.5', 'dir')
m2c_opts.cudaDir = {'/Developer/NVIDIA/CUDA-7.5'};
else
error('m2c:cuda_dir', ...
['Root directory of CUDA must be given after the ' ...
'-cuda flag or be specified by environment variable CUDA_PATH.\n']);
end
end
m2c_opts.withCuda = true;
m2c_opts.cudaInc = {['-I' m2c_opts.cudaDir{1} '/include']};
if exist([m2c_opts.cudaDir{1} '/lib64'], 'dir')
cudaLibDir = [m2c_opts.cudaDir{1} '/lib64'];
else
cudaLibDir = [m2c_opts.cudaDir{1} '/lib'];
end
m2c_opts.cudaLibs = {['-L' cudaLibDir], '-lcublas', '-lcusparse', '-lcudart'};
if ~isequal(opt, '-cuda')
m2c_opts.withNvcc = true;
end
case {'-mkl', 'mkl-seq', '-mkl-iomp', '-mkl-omp'}
if i<last_index && varargin{i+1}(1) == '{'
m2c_opts.mklDir = eval(varargin{i+1});
i = i + 1;
elseif i<last_index-1 && varargin{i+1}(1) ~= '-' && ~isequal(varargin{i+1}(2), '-args')
m2c_opts.mklDir = varargin(i+1);
i = i + 1;
else
if ~isempty(getenv('MKLROOT'))
m2c_opts.mklDir = {getenv('MKLROOT')};
elseif isunix && exist('/opt/intel/mkl', 'dir')
m2c_opts.mklDir = {'/opt/intel/mkl'};
else
error('m2c:mkl_dir', ...
['Root directory of MKL must be given after the ' ...
'-mkl flag or be specified by environment variable MKLROOT. \n' ...
'Intel MKL and TBB can be downloaded for free at ' ...
'https:
end
end
if m2c_opts.mklDir{1}(end)=='/' || m2c_opts.mklDir{1}(end)=='\'
m2c_opts.mklDir{1}(end) = [];
end
m2c_opts.withMKL = true + contains(opt, 'omp');
m2c_opts.mklInc = {['-I' m2c_opts.mklDir{1} '/include']};
m2c_opts.mklLibs = {dylibdir([m2c_opts.mklDir{1} '/lib']), ...
'-lmkl_intel_lp64', '-lmkl_core'};
if isequal(opt, '-mkl-iomp')
m2c_opts.mklLibs = [m2c_opts.mklLibs, ...
dylibdir([fileparts(m2c_opts.mklDir{1}) '/lib']), ...
'-lmkl_intel_thread', '-liomp5'];
else
m2c_opts.mklLibs = [m2c_opts.mklLibs, '-lmkl_sequential'];
end
case '-petsc'
if i<last_index && varargin{i+1}(1) == '{'
m2c_opts.petscDir = eval(varargin{i+1});
i = i + 1;
elseif i<last_index-1 && varargin{i+1}(1) ~= '-' && ~isequal(varargin{i+1}(2), '-args')
m2c_opts.petscDir = varargin(i+1);
i = i + 1;
else
if ~isempty(getenv('PETSC_DIR')) && ~isempty(getenv('PETSC_ARCH'))
m2c_opts.petscDir = {[getenv('PETSC_DIR') '/' getenv('PETSC_ARCH')]};
elseif ~isempty(getenv('PETSC_DIR'))
m2c_opts.petscDir = {getenv('PETSC_DIR')};
elseif exist('/usr/lib/petsc', 'dir')
% Set default path for Ubuntu
m2c_opts.petscDir = {'/usr/lib/petsc'};
elseif length(varargin) >1
warning('m2c:petsc_dir', ...
['Root directory of PETSc must be given after the ' ...
'-petsc flag or be specified by environment variable PETSC_DIR.\n']);
else
% This is for checking MPI path
break;
end
end
petscvariables = [m2c_opts.petscDir{1} '/lib/petsc/conf/petscvariables'];
if ~exist(petscvariables, 'file')
warning('m2c:petsc_dir', ...
['Could not locate file lib/petsc/conf/petscvariables ' ...
'under the given petsc directory %s.\n'], m2c_opts.petscDir{1});
end
if ~exist([m2c_opts.petscDir{1} '/lib/libpetsc.so'], 'file') && ...
~exist([m2c_opts.petscDir{1} '/lib/libpetsc_real.so'], 'file') && ...
~exist([m2c_opts.petscDir{1} '/lib/libpetsc.dylib'], 'file')
warning('m2c:petsc', ...
'PETSc must be built as a shared library in order to be used in MATLAB.\n');
end
[PCC, CXX, INC, CFLAGS, CXXFLAGS, LIBEXT] = obtain_petscCC(petscvariables);
if isempty(PCC)
warning('m2c:petsc_dir', ...
['Could find the definition of PCC in lib/petsc/conf/petscvariables ' ...
'under the given petsc directory %s.\n'], m2c_opts.petscDir{1});
end
m2c_opts.petscCC = {PCC};
m2c_opts.petscCXX = {CXX};
m2c_opts.petscCFLAGS = {CFLAGS};
m2c_opts.petscCXXFLAGS = {CXXFLAGS};
mpetscInc = [fileparts(which('load_petsc.m')), '/include'];
m2c_opts.petscInc = {[INC ' -I' mpetscInc]};
m2c_opts.petscLibs = {dylibdir([m2c_opts.petscDir{1}  '/lib']), ['-lpetsc' LIBEXT]};
m2c_opts.withPetsc = true;
case '-blas'
if i<last_index && varargin{i+1}(1) == '{'
m2c_opts.blaskLibs = eval(varargin{i+1});
i = i + 1;
elseif i<last_index-1 && varargin{i+1}(1) ~= '-' && ~isequal(varargin{i+1}(2), '-args')
m2c_opts.blaskLibs = varargin(i+1);
i = i + 1;
elseif ismac
m2c_opts.blasLibs = {'-lblas'};
else
m2c_opts.blasLibs = {'-lmwblas'};
end
m2c_opts.withBlas = true;
case '-lapack'
if i<last_index && varargin{i+1}(1) == '{'
m2c_opts.lapackLibs = eval(varargin{i+1});
i = i + 1;
elseif i<last_index-1 && varargin{i+1}(1) ~= '-' && ~isequal(varargin{i+1}(2), '-args')
m2c_opts.lapackLibs = varargin(i+1);
i = i + 1;
elseif ismac
m2c_opts.lapackLibs = {'-llapack', '-lblas'};
else
m2c_opts.lapackLibs = {'-lmwlapack', '-lmwblas'};
end
m2c_opts.withLapack = true;
case '-omp'
m2c_opts.withOMP = true;
if i<last_index && varargin{i+1}(1) == '{'
m2c_opts.ompLibs = eval(varargin{i+1});
i = i + 1;
elseif i<last_index-1 && varargin{i+1}(1) ~= '-' && ~isequal(varargin{i+1}(2), '-args')
m2c_opts.ompLibs = varargin(i+1);
i = i + 1;
else
m2c_opts.([opt(2:end) 'Libs']) = {' '};
end
case {'-mpi'}
m2c_opts.withMPI= true;
if i<last_index && varargin{i+1}(1) == '{'
m2c_opts.mpiLibs = eval(varargin{i+1});
i = i + 1;
elseif i<last_index-1 && varargin{i+1}(1) ~= '-' && ~isequal(varargin{i+1}(2), '-args')
m2c_opts.mpiLibs = varargin(i+1);
i = i + 1;
else
m2c_opts.mpiLibs = {' '};
end
if ~isequal(m2c_opts.petscCC, {''})
petsc_opts = m2c_opts;
else
petsc_opts = proc_options('-petsc');
end
if ~isempty(petsc_opts.petscCC)
m2c_opts.mpiCC = petsc_opts.petscCC;
m2c_opts.mpiCXX = petsc_opts.petscCXX;
else
m2c_opts.mpiCC = {'mpicc'};
m2c_opts.mpiCXX = {'mpicc'};
end
mpiInc = [fileparts(which('startup_mmpi.m')), '/include'];
m2c_opts.mpiInc = {['-I' mpiInc]};
case '-config'
if i<last_index && varargin{i+1}(1) == '{'
m2c_opts.codegenConfig = eval(varargin{i+1});
i = i + 1;
elseif i<last_index-1 && varargin{i+1}(1) ~= '-' && ~isequal(varargin{i+1}(2), '-args')
m2c_opts.codegenConfig = varargin(i+1);
i = i + 1;
else
error('m2c:wrong_argument', ...
'Argument %s requires a cell-array argument after it.\n', opt);
end
case '-v'
m2c_opts.verbose = true;
case '-q'
m2c_opts.quiet = true;
case '-m'
warning('Option -m is suppresed.');
case {'-c++','-cxx','-lang:c++'}
m2c_opts.useCpp = true;
m2c_opts.suf = 'cpp';
case {'-h', '-?', '--help'}
m2c_opts = [];
return;
otherwise
if isequal(opt, '-args')
if func_index==0 || func_index~=i-1
error('m2c:InvalidArgument', '-args must follow a function name.\n');
elseif i==last_index || varargin{i+1}(1)~='{'
error('m2c:InvalidArgument', '-args must be followed by a cell array.\n');
else
if isempty(m2c_opts.codegenArgs)
m2c_opts.codegenArgs = [opt ' ' varargin{i+1}];
else
m2c_opts.codegenArgs = sprintf('%s ', ...
m2c_opts.codegenArgs, opt, varargin{i+1});
end
i = i + 1;
end
elseif ~isempty(opt) && opt(1)~='-'
if func_index==0
func_index = i;
if exist(opt, 'file') || exist([opt '.m'], 'file')
m2c_opts.funcName = {opt};
else
error('m2c:InvalidFileName', 'Function %s could not be found.\n', opt);
end
elseif ~isempty(m2c_opts.codegenArgs) || func_index==i-3 && ...
i<last_index-2 && isequal(varargin{i+1}, '-args')
func_index = i;
m2c_opts.codegenArgs = sprintf('%s ', m2c_opts.codegenArgs, opt);
else
error('m2c:wrong_argument', 'Unexpected string %s.\n', opt);
end
elseif length(opt)>=2 && (isequal(opt(1:2), '-D') || isequal(opt(1:2), '-I'))
m2c_opts.cppflags = [m2c_opts.cppflags, opt];
elseif length(opt)>=2 && isequal(opt(1:2), '-L')
m2c_opts.libs = [m2c_opts.libs, dylibdir(opt(3:end))];
elseif length(opt)>=2 && isequal(opt(1:2), '-l')
m2c_opts.libs = [m2c_opts.libs, opt];
else
error('m2c:wrong_argument', 'Unrecognized argument %s.\n', opt);
end
end
i = i +1;
end
for i=1:length(names)
if iscell(m2c_opts.(names{i})) && isequal(m2c_opts.(names{i}){1}, '')
m2c_opts.(names{i})(1) = [];
elseif iscell(m2c_opts.(names{i})) && isequal(m2c_opts.(names{i}){1}, ' ')
m2c_opts.(names{i})(1) = {''};
end
end
if ~isempty(m2c_opts.funcName)
mpath = fileparts(which(m2c_opts.funcName{1}));
if exist([mpath '/include'], 'dir')
m2c_opts.cppflags = [m2c_opts.cppflags, ['-I' mpath '/include']];
end
end
if isoctave
% When running in octave, always skip codegen
m2c_opts.skipcg = true;
end
if m2c_opts.optimLevel<0
m2c_opts.optimLevel=int32(~m2c_opts.debugInfo);
end
if ~m2c_opts.debugInfo
m2c_opts.m2cpath = [m2c_opts.m2cpath ['-I ' m2croot__ '/opts/No_debug']];
end
if ~isempty(m2c_opts.valgrind) && ~m2c_opts.withCuda
m2c_opts.chkMem = true;
end
m2c_opts.genExe = m2c_opts.genExe || ~isempty(m2c_opts.gdb) || ...
~isempty(m2c_opts.ddd) || ~isempty(m2c_opts.valgrind) || ...
~isempty(m2c_opts.gprof) || ~isempty(m2c_opts.gcov);
if m2c_opts.genExe && ~isunix()
if ~m2c_opts.quiet
fprintf('Warning: Building executable is not supported on Windows PCs\n');
end
m2c_opts.genExe = false;
end
if m2c_opts.withCuda
m2c_opts.cppflags = [m2c_opts.cppflags '-DM2C_CUDA=1'];
if m2c_opts.withNvcc
m2c_opts.m2cpath = [m2c_opts.m2cpath ['-I ' m2croot__ '/opts/cuda']];
end
if ~m2c_opts.withNvcc && (m2c_opts.withPetsc || m2c_opts.withMPI)
warning('m2c:WarnCUDAwithMPI', ['You have enabled CUDA together with PETSc/MPI.\n', ...
'Mixed compilation of CUDA with PETSc/MPI is not supported. Compilation \n', ...
'and linking will be performeed using mpicc instead of nvcc. No CUDA kernel\n', ...
'function will be generated. Use -nvcc instead of -cuda to force compilation with nvcc.'\n]);
end
if m2c_opts.chkMem
warning('m2c:WarnCUDAwithMemChk', 'Memory check is not suported with CUDA. Disabled.\n');
m2c_opts.chkMem = false;
end
end
if m2c_opts.withNvcc && (m2c_opts.withPetsc || m2c_opts.withMPI || m2c_opts.withOMP)
error('m2c:NVCCWrontOption', 'You cannot mix NVCC with PETSc, MPI, or OpenMP.');
end
if m2c_opts.withMKL
% When using MKL, diable BLAS and its libraries
m2c_opts.withBlas = false;
m2c_opts.blasLibs = {};
m2c_opts.cppflags = [m2c_opts.cppflags '-DM2C_MKL=1'];
if m2c_opts.withOMP && m2c_opts.withMKL>1 && isempty(m2c_opts.cc)
warning('m2c:WarnMKLwithOMP', ['You are using OpenMP together with Intel OpenMP in MKL.\n', ...
'This may cause conflicts! See https:
'-for-calling-intelr-mkl-routines-from-multi-threaded-applications for recommended settings.\n', ...
'To disable this message, explicitly use -cc to specify an Intel compiler.']);
end
elseif m2c_opts.withBlas
m2c_opts.cppflags = [m2c_opts.cppflags '-DM2C_BLAS=1'];
end
if m2c_opts.optimLevel && isempty(m2c_opts.gdb) && isempty(m2c_opts.ddd)
m2c_opts.cppflags = [m2c_opts.cppflags '-DNDEBUG'];
end
if m2c_opts.withOMP
m2c_opts.m2cpath = [m2c_opts.m2cpath ['-I ' m2croot__ '/opts/omp']];
end
if m2c_opts.withMPI
m2c_opts.cppflags = [m2c_opts.cppflags '-DM2C_MPI=1'];
end
end
function build_exe(cpath, func)
command = [cpath 'build_' func '_exe.m'];
clear(command);
if exist(command, 'file'); run(command); end
end
function [PCC, CXX, INC, CFLAGS, CXXFLAGS, LIBEXT] = obtain_petscCC(filename)
% Obtian the PCC and CXX commands from the petscvariables file
str = readFile(filename);
pat = '\nPCC\s*=\s*([^\n]+)\n';
def = regexp(str, pat, 'match', 'once');
if ~isempty(def)
PCC = strtrim(regexprep(def, pat, '$1'));
else
PCC = '';
end
if nargout>1
pat = '\nCXX\s*=\s*([^\n]+)\n';
def = regexp(str, pat, 'match', 'once');
if ~isempty(def)
CXX = strtrim(regexprep(def, pat, '$1'));
else
CXX = '';
end
end
if nargout>2
if exist('petscroot', 'file')
INC = ['-I' petscroot '/include '];
else
INC = '';
end
pat = 'PETSC_CC_INCLUDES\s*=\s*([^\n]+)\n';
def = regexp(str, pat, 'match', 'once');
if ~isempty(def)
INC = [INC strtrim(regexprep(def, pat, '$1'))];
end
end
if nargout>3
CFLAGS = '';
pat = '\nCC_FLAGS\s*=\s*([^\n]+)\n';
def = regexp(str, pat, 'match', 'once');
if ~isempty(def)
CFLAGS = strtrim(regexprep(def, pat, '$1'));
end
end
if nargout>4
CXXFLAGS = '';
pat = 'CXX_FLAGS\s*=\s*([^\n]+)\n';
def = regexp(str, pat, 'match', 'once');
if ~isempty(def)
CXXFLAGS = strtrim(regexprep(def, pat, '$1'));
end
end
if nargout>5
LIBEXT = '';
pat = 'PETSC_LIB_EXT\s*=\s*([^\n]+)\n';
def = regexp(str, pat, 'match', 'once');
if ~isempty(def)
LIBEXT = strtrim(regexprep(def, pat, '$1'));
end
end
end
function str = dylibdir(dir)
str = ['-L' dir];
if ~isoctave || ~strncmp(OCTAVE_VERSION, '3', 1)
% Use rpath only for Octave 4.0 or higher
if ismac
concat = ',';
else
concat = '=';
end
str = [str, ' -Wl,-rpath' concat dir];
end
end
