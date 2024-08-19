function OMP_begin_for %#codegen
%OMP_begin_for specifies that the iterations of the loop immediately
% following it must be executed in parallel by the team.
%
%     OMP_begin_for
% This assumes a parallel region has already been initiated by
% OMP_begin_parallel. Otherwise it executes in serial on a single processor.
%
% Use OMP_clause_nowait to avoid the implied barrier at the end of the 
% for directive. 
% This directive must be paired with OMP_end_for
%
% See also OMP_end_for, OMP_clause_schedule, OMP_clause_ordered, 
%     OMP_clause_private, OMP_clause_firstprivate, OMP_clause_lastprivate, 
%     OMP_clause_reduction, OMP_clause_schedule, OMP_clause_collapse, 
%     OMP_clause_nowait
if ~strcmp(coder.target,'rtw'); return; end
coder.inline('always')
coder.allowpcode('plain')
coder.ceval('#pragma momp for');
coder.ceval('#{for');
