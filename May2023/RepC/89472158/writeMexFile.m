function writeMexFile(funcname, mpath, cpath, m2c_opts, parmode)
% TODO: Implement support for varargin, varargout, cell arrays, complex
%       numbers, global variables, and function pointers.
if isequal(parmode, 'cuda-kernel')
altapis = {[funcname '_cuda0'], [funcname '_cuda1'], [funcname '_cuda2']};
else
altapis = [funcname, strtrim(strrep(regexp(m2c_opts.codegenArgs, ...
'(\w+)\s+-args', 'match'), ' -args', ''))];
end
% Write out _mex file
if m2c_opts.withNvcc
outMexFile = [cpath funcname '_mex.cpp'];
else
outMexFile = [cpath funcname '_mex.' m2c_opts.suf];
end
[~, signature] = ckSignature(m2c_opts, 'codegen');
if cpath(1) == '/' || cpath(1) == '\'
fname = [funcname '_mex.' m2c_opts.suf];
else
fname = outMexFile;
end
if isequal(parmode, 'cuda-kernel')
otherheader = '#include "cuda4m.h"';
elseif isequal(parmode, 'omp-kernel')
otherheader = '#include "omp.h"';
else
otherheader = '';
end
if exist([cpath funcname '_types.h'], 'file')
types_header = ['#include "' funcname '_types.h"'];
else
types_header = '';
end
filestr = sprintf('%s\n', ...
'', '', ...
'#include "mex.h"',  ...
'#if !defined(MATLAB_MEX_FILE) && defined(printf)', ...
'#undef printf', ...
'#endif',  ...
'', ...
['#include "' funcname '.h"'], types_header, ...
'#include "m2c.c"', otherheader, ...
'');
if m2c_opts.withNvcc
filestr = sprintf('%s%s\n', filestr, ...
'#include "lib2mex_helper.cpp"');
else
filestr = sprintf('%s%s\n', filestr, ...
'#include "lib2mex_helper.c"');
end
structDefs = struct();
apiStr = '';
% Parse input arguments from C files.
alt_nlhs = zeros(length(altapis),1);
alt_nrhs = zeros(length(altapis),1);
infofile = [cpath 'codeInfo.mat'];
if ~exist(infofile, 'file')
error('m2c:writeMexFile', 'Cannot locate codeInfo file for function %s.', ...
funcname);
end
%% Read in codeInfo file for further processing
load(infofile,'codeInfo');
prototypes = [codeInfo.OutputFunctions.Prototype];
inports = codeInfo.Inports;
outports = codeInfo.Outports;
args = strtrim(strrep(regexp(m2c_opts.codegenArgs, '-args\s+{[^}]*}', 'match'), '-args ', ''));
for i=1:length(args)
args{i} = eval(args{i});
end
if isequal(parmode, 'cuda-kernel')
% Parse input arguments from C files.
[vars, ret, nlhs, nrhs, structDefs, SDindex, pruned_vars] = ...
parse_cgfiles(funcname, funcname, prototypes(1), args{1}, inports, outports, cpath, structDefs);
vars_options = cuda_control_variables(nrhs);
if ~isempty(ret)
error('m2c:CudaKernelHasReturn', ...
['You have a scalar output ' ret.name ' for the CUDA kernel function. You must\n' ...
'add it to the input arguments so that the variable is passed by reference']);
end
for i=1:length(altapis)
if i>1
nrhs = nrhs + 1;
vars = [vars; vars_options(i-1)]; %#ok<AGROW>
end
[func, structDefs] = printApiFunction(funcname, altapis{i}, vars, ret, ...
structDefs, SDindex, pruned_vars, m2c_opts.timing, parmode, m2c_opts.debugInfo);
apiStr = sprintf('%s\n%s', apiStr, func);
alt_nlhs(i) = nlhs; alt_nrhs(i) = nrhs;
end
else
inport_start = 1;
outport_start = 1;
for i=1:length(altapis)
% Check existence of the files
mfile = [mpath altapis{i} '.m'];
if ~exist(mfile, 'file') && exist([mpath 'codegen/' altapis{i} '.m'], 'file')
mfile = [mpath 'codegen/' altapis{i} '.m'];
end
if isempty(mfile)
error('m2c:parse_cgfile', 'Cannot find MATLAB file %s', funcname);
end
m_args = extractMfileArgs(mfile);
% Parse input and output arguments from codenInfo.
inport_next = inport_start + length(args{i});
outport_next = outport_start + length(m_args.output);
if length(altapis)>1
for j=inport_start:inport_next-1
assert(isempty(inports(j).Implementation) || ...
~isprop(inports(j).Implementation, 'Identifier') || ...
(strcmp(inports(j).Implementation.Identifier, ...
[inports(j).GraphicalName '_' altapis{i}]) || ...
strcmp(inports(j).Implementation.Identifier(3:end), ...
[inports(j).GraphicalName '_' altapis{i}])));
end
for j=outport_start:outport_next-1
assert(isempty(outports(j).Implementation) || ...
~isprop(outports(j).Implementation, 'Identifier') || ...
(strcmp(outports(j).Implementation.Identifier, ...
[outports(j).GraphicalName '_' altapis{i}]) || ...
strcmp(outports(j).Implementation.Identifier(3:end), ...
[outports(j).GraphicalName '_' altapis{i}])));
end
end
[vars, ret, nlhs, nrhs, structDefs, SDindex, pruned_vars] = ...
parse_cgfiles(funcname, altapis{i}, prototypes(i), args{i}, ...
inports(inport_start:inport_next-1), ...
outports(outport_start:outport_next-1), cpath, structDefs);
if inport_next <= length(inports)
inport_start = inport_next;
end
if outport_next <= length(outports)
outport_start = outport_next;
end
if ~isempty(SDindex)
fprintf(2, ['m2c Info: Codegen generated a StackData object "' ...
vars(SDindex).cname '" of type "' vars(SDindex).type '". ' ...
'This probably indicates that you have some large, ', ...
'fixed-size local buffers in some subroutines ' ...
'that Codegen grouped into an object. See "' ...
cpath funcname '_types.h" ' , ...
'for the definition and content of the object.\n']);
end
[func, structDefs] = printApiFunction(funcname, altapis{i}, vars, ret, ...
structDefs, SDindex, pruned_vars, m2c_opts.timing, parmode, m2c_opts.debugInfo);
apiStr = sprintf('%s\n%s', apiStr, func);
alt_nlhs(i) = nlhs; alt_nrhs(i) = nrhs;
end
end
% Print out helper functions for marshalling in/out data
fields = fieldnames(structDefs);
for i = 1:length(fields)
f = structDefs.(fields{i});
filestr = sprintf('%s\n%s\n', filestr, ...
[f.marshallinFunc, f.marshallinArrayFunc, ...
f.marshallinConstFunc, f.marshallinConstArrayFunc, ...
strtrim(f.preallocFunc), strtrim(f.preallocArrayFunc), ...
f.marshalloutFunc, f.marshalloutArrayFunc, ...
strtrim(f.destroyFunc), strtrim(f.destroyArrayFunc)]);
end
% Print out API and Mex functions.
filestr = sprintf('%s\n', filestr, apiStr, ...
printMexFunction(altapis, alt_nlhs, alt_nrhs));
writeFile(outMexFile, filestr);
end
function [str, structDefs] = printApiFunction(funcname, altname, vars, ...
ret, structDefs, SDindex, pruned_vars, timing, parmode, debugMode)
% Print an API function into a string for the function with the given
% numbers of input and output arguments.
if ~isempty(ret)
retval = [ret.cname ' = '];
else
retval = '';
end
vars_ret = [vars; ret];
if isempty(SDindex)
SDname = '';
else
SDname = ['&' vars(SDindex).cname];
end
iscuda = strncmp(parmode, 'cuda', 4);
[marshallin_substr, structDefs, useNDims, hasCuError_in] = marshallin ...
(vars_ret, altname, iscuda, structDefs);
[marshallout_substr, structDefs, maxDim, hasCuError_out] = marshallout ...
(vars_ret, pruned_vars, iscuda, structDefs);
hasCuError = hasCuError_in || hasCuError_out;
% Preallocate StackData object
if ~isempty(SDindex)
var = vars(SDindex);
sf = vars(SDindex).subfields;
prealloc_substr = ['    ' char(10)]; %#ok<*CHARTEN>
if ~iscuda
prealloc_substr = sprintf('%s%s\n', prealloc_substr, ...
['    ' var.cname '.' sf.cname ' = (' sf.type '*)mxMalloc(sizeof(' sf.type '));']);
else
prealloc_substr = sprintf('%s%s\n%s\n', prealloc_substr, ...
['    _ierr = cudaMalloc((void **)&' var.cname '.' sf.cname ', sizeof(' sf.type '));'], ...
'    CHKCUERR(_ierr, "cudaMalloc");');
end
else
prealloc_substr = '';
end
str = sprintf('%s\n', ...
['static void __' altname '_api(mxArray **plhs, const mxArray ** prhs) {'], ...
declareVars(vars, ret, timing, iscuda, useNDims, maxDim, hasCuError), ...
[marshallin_substr, prealloc_substr], '', ...
'    ');
if ~isempty(SDname)
str = sprintf('%s%s\n', str, ...
['    ' funcname '_initialize(' SDname ');']);
end
if ~isempty(timing)
str = sprintf('%s%s\n', str, '    _timestamp = M2C_wtime();');
end
isomp_kernel = isequal(parmode, 'omp-kernel');
if isomp_kernel
str = sprintf('%s#pragma omp parallel\n    {\n', str);
if debugMode
str = sprintf('%s#pragma omp single\n        %s\n\n', str, ...
'M2C_printf("Running with %d OpenMP threads\n", omp_get_num_threads());');
end
str = [str '    '];
end
str = sprintf('%s%s\n', str, ...
['    ' retval altname '(' list_args(vars, iscuda) ');']);
if isomp_kernel
str = sprintf('%s    }\n', str);
end
if ~isempty(timing)
str = sprintf('%s%s\n%s\n', str, ...
'    _timestamp = M2C_wtime() - _timestamp;', ...
['    printf("Function ' funcname ' completed in %g seconds.\n", _timestamp);']);
end
if ~isempty(SDindex)
var = vars(SDindex);
sf = vars(SDindex).subfields;
destroy_substr = ['    ' char(10)];
if ~iscuda
destroy_substr = sprintf('%s%s\n', destroy_substr, ...
['    mxFree(' var.cname '.' sf.cname ');']);
else
destroy_substr = sprintf('%s%s\n%s\n', destroy_substr, ...
['    _ierr = cudaFree(' var.cname '.' sf.cname ');'], ...
'    CHKCUERR(_ierr, "cudaFree");');
end
else
destroy_substr = '';
end
if ~isempty(SDname)
str = sprintf('%s%s\n', str, ...
['    ' funcname '_terminate();']);
end
str = sprintf('%s\n', str, ...
[marshallout_substr, destroy_substr], '}');
if ~isempty(SDindex)
end
% Remove two consecutive empty lines
while ~isempty(regexp(str, '\n\n\n', 'once'))
str = regexprep(str, '(\n\n)\n', '$1');
end
end
function str = printMexFunction(altapis, alt_nlhs, alt_nrhs)
% Print into a string a mexFunction for the function with the given
% numbers of input and output arguments.
str = 'void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {';
if max(alt_nlhs)>1
str = sprintf('\n%s', ...
str, '    ', ...
['    mxArray *outputs[' int2str(max(alt_nlhs)) '] = {' ...
repmat('NULL, ', 1, max(alt_nlhs)-1) 'NULL};'], ...
'    int i;', ...
'    int nOutputs = (nlhs < 1 ? 1 : nlhs);', '');
end
haselse = '';
for i=1:length(altapis)
funcname = altapis{i}; nlhs = alt_nlhs(i); nrhs = alt_nrhs(i);
str = sprintf('\n%s', str, ...
['    ' haselse 'if (nrhs == ' int2str(nrhs) ') {'], ...
['        if (nlhs > ' int2str(nlhs) ')'], ...
['            mexErrMsgIdAndTxt("' funcname ':TooManyOutputArguments",'], ...
['                "Too many output arguments for entry-point ' funcname '.\n");']);
if max(alt_nlhs)<=1
str = sprintf('\n%s', str, ...
'        ', ...
['        __' funcname '_api(plhs, prhs);'], '    }');
else
str = sprintf('\n%s', str, ...
'        ', ...
['        __' funcname '_api(outputs, prhs);'], '    }');
end
haselse = 'else ';
end
str = sprintf('\n%s', str, ...
'    else', ...
['        mexErrMsgIdAndTxt("' altapis{1} ':WrongNumberOfInputs",'], ...
['            "Incorrect number of input variables for entry-point ' altapis{1} '.");']);
if max(alt_nlhs)>1
str = sprintf('\n%s', str, '', ...
'    ', ...
'    for (i = 0; i < nOutputs; ++i) {', ...
'        plhs[i] = outputs[i];', ...
'    }');
end
str = sprintf('\n%s', str, '}');
% Remove two consecutive empty lines
while ~isempty(regexp(str, '^\n\n', 'once'))
str = regexprep(str, '^(\n)\n', '$1');
end
while ~isempty(regexp(str, '\n\n\n', 'once'))
str = regexprep(str, '(\n\n)\n', '$1');
end
end
function str = declareVars(vars, ret, timing, iscuda, useNDims, maxDim, hasCuError)
% Produce variable declarations
decl_args = '';
decl_tmps = '';
for i=1:length(vars)
var = vars(i);
if strncmp(var.type, 'emxArray_', 9) || ~isempty(var.subfields) && ...
isequal(var.size, 1) && ~var.vardim
if ~iscuda
% Put emxArray on stack
decl_args = sprintf('%s    %-20s %s;\n', decl_args, ...
var.type, var.cname);
else
decl_args = sprintf('%s    %-20s*%s; \n', decl_args, ...
var.type, var.cname);
end
elseif isempty(var.iindex) && isempty(var.oindex)
% A variable size array, a structure, or a string
if isempty(var.modifier)
error('var/modifier');
end
assert(~isempty(var.modifier));
if ~iscuda
decl_args = sprintf('%s    %-20s %s[%d];\n', decl_args, ...
var.type, var.cname, prod(var.size));
else
decl_args = sprintf('%s    %-20s*%s; \n', decl_args, ...
var.type, var.cname);
end
elseif ~iscuda || isempty(var.modifier)
if ~isempty(var.modifier)
decl_args = sprintf('%s    %-20s*%s;\n', decl_args, ...
var.type, var.cname);
else
decl_args = sprintf('%s    %-20s %s;\n', decl_args, ...
var.type, var.cname);
end
else
decl_args = sprintf('%s    %-20s*%s; \n', decl_args, ...
var.type, var.cname);
end
end
if ~isempty(ret)
decl_args = sprintf('%s    %-20s %s%s;\n', decl_args, ...
ret.type, ret.modifier, ret.cname);
end
if ~isempty(timing)
% declare variables for timing
decl_tmps = sprintf('%s    %-20s _timestamp;\n', decl_tmps, 'double');
end
if useNDims
decl_tmps = sprintf('%s    %-20s _nDims;\n', decl_tmps, 'int');
end
if maxDim
decl_tmps = sprintf('%s\n', decl_tmps, ...
sprintf('    %-20s %s;', 'int', ['_dims[' num2str(maxDim) ']']));
end
if iscuda
if hasCuError
decl_tmps = sprintf('%s    %-20s _ierr;\n', decl_tmps, 'cudaError_t');
end
end
str = sprintf('%s', decl_args, decl_tmps);
end
function str = list_args(vars, iscuda)
% List the arguments for calling the main entrance
str = '';
for i=1:length(vars)
if vars(i).isemx
if ~iscuda && strncmp(vars(i).type, 'emxArray_', 9)
str = sprintf('%s, &%s', str, vars(i).cname);
else
str = sprintf('%s, %s', str, vars(i).cname);
end
elseif ~isempty(vars(i).subfields)
if ~iscuda && ~isempty(vars(i).modifier) && prod(vars(i).size)==1 && ~vars(i).vardim
modifier='&';
else
modifier='';
end
str = sprintf('%s, %s%s', str, modifier, vars(i).cname);
else
str = sprintf('%s, %s', str, vars(i).cname);
end
end
str = str(3:end);
end
function opts = cuda_control_variables(nrhs)
% Control varilables for CUDA
nthreads = struct('cname', '_nthreads', 'mname', 'nthreads', ...
'type', 'int32_T', 'basetype', 'int32_T', 'structname', '', ...
'modifier', '', 'isconst', false, 'iscomplex', false, 'subfields', [], ...
'isemx', false, 'size', 1, 'vardim', 0, 'sizefield', [], ...
'iindex', nrhs+1, 'oindex', []);
threadsPB = struct('cname', '_threadsPB', 'mname', 'threadsPB', ...
'type', 'int32_T', 'basetype', 'int32_T', 'structname', '', ...
'modifier', '', 'isconst', false, 'iscomplex', false, 'subfields', [], ...
'isemx', false, 'size', 1, 'vardim', 0, 'sizefield', [], ...
'iindex', nrhs+2, 'oindex', []);
opts = [nthreads, threadsPB];
end
