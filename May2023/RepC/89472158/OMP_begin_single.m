function OMP_begin_single %#codegen
%OMP_begin_single  specifies that the enclosed code is to be executed
%     by only one thread in the team.
%
%     OMP_begin_single
%
% This directive should be paired with OMP_end_single. Use
% OMP_clause_nowait to avoid the implied barrier at the end of the 
% for directive. 
%
% See also OMP_end_single, OMP_begin_single, OMP_clause_nowait
if ~strcmp(coder.target,'rtw'); return; end
coder.inline('always')
coder.allowpcode('plain')
coder.ceval('#pragma momp single');
coder.ceval('#{single');
