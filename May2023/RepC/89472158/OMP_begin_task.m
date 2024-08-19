function OMP_begin_task %#codegen
%OMP_begin_task  defines an explicit task, which may be executed by the 
% encountering thread, or deferred for execution by any other thread in the team.
%
%     OMP_begin_task
%
% This directive does not accept clauses. It must be paired with OMP_end_task.
% 
% Note: This corresponds to a new construct with OpenMP 3.0.
%
% See also OMP_end_task
coder.inline('always')
coder.allowpcode('plain')
if ~strcmp(coder.target,'rtw'); return; end
coder.ceval('#pragma momp task');
coder.ceval('#{task');
