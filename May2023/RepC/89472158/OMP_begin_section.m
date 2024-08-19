function OMP_begin_section %#codegen
%Declares a section to be executed once by a thread in the team.
%
%     OMP_begin_section
%
% This directive should be paired with OMP_end_section.
%
% See also OMP_end_section, OMP_begin_sections, OMP_begin_parallel, 
%      OMP_clause_nowait
if ~strcmp(coder.target,'rtw'); return; end
coder.inline('always')
coder.allowpcode('plain')
coder.ceval('#pragma momp section');
coder.ceval('#{section');
