function OMP_begin_sections %#codegen
%OMP_begin_sections It specifies that the enclosed section(s) of code are
% to be divided among the threads in the team.
%
%     OMP_begin_sections
%
% This directive should be paired with OMP_end_sections. Use 
% OMP_clause_nowait to avoid the implied barrier at the end of the
% sections directive.
%
% See also OMP_end_sections, OMP_begin_section, OMP_begin_parallel,
%      OMP_clause_nowait
if ~strcmp(coder.target,'rtw'); return; end
coder.inline('always')
coder.allowpcode('plain')
coder.ceval('#pragma momp sections');
coder.ceval('#{sections');
