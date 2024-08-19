function OMP_taskwait %#codegen
%OMP_taskwait  pecifies a wait on the completion of child tasks 
%     generated since the beginning of the current task.
%
% This directive does not accept clauses.
% It is new with OpenMP 3.0.
coder.inline('always')
coder.allowpcode('plain')
coder.allowpcode('plain')
if ~strcmp(coder.target,'rtw'); return; end
coder.ceval('#pragma momp taskwait');
