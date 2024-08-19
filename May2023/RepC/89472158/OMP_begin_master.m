function OMP_begin_master %#codegen
%OMP_begin_master  marks the beginning of the OpenMP master block.
%
% It specifies a region that is to be executed only by the master thread of
% the team. All other threads on the team skip this section of code.
%
% This directive does not accept clauses. No implied barrier exists on
% either entry to or exit from the master section.
% It must be paired with OMP_end_master.
%
% See also OMP_end_master
if ~strcmp(coder.target,'rtw'); return; end
coder.inline('always')
coder.allowpcode('plain')
coder.ceval('#pragma momp master');
coder.ceval('#{master');
