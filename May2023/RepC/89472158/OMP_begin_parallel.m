function OMP_begin_parallel(nthreads) %#codegen
%OMP_begin_parallel  Declares a parallel region.
%
%     OMP_begin_parallel
%     OMP_begin_parallel(nthreads)
%
% A parallel region is a block of code that will be executed by multiple
% threads. This is the fundamental OpenMP parallel construct. This directive
% should be paired with OMP_end_single. An implied barrier exists at the end
% of a parallelized statement block.
%
% The statement between OMP_begin_parallel and OMP_end_parallel
% should be a sequence of function calls to functions that
% explicitly have specified "coder.inline('never')".
%
% See also OMP_end_parallel, OMP_clause_num_threads
if ~strcmp(coder.target,'rtw'); return; end
coder.inline('always');
coder.allowpcode('plain')
if nargin>0
coder.ceval('#pragma momp parallel default(shared) num_threads', nthreads);
else
coder.ceval('#pragma momp parallel default(shared)');
end
coder.ceval('#{parallel');
