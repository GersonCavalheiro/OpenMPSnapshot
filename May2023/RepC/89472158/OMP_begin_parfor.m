function OMP_begin_parfor %#codegen
%OMP_begin_parfor combines parallel and for.
%
%     OMP_begin_parfor
%
% In addition, you should use OMP_clause_shared, OMP_clause_private,
% OMP_clause_first_private etc. to declare all the variables.
% The directive uses default(none) by default, but you can overwrite it
% with OMP_clause_default('shared').
%
% An implied barrier exists at the end of a parfor statement block.
% This directive should be paired with OMP_end_parfor.
%
% See also OMP_end_parfor, OMP_begin_parallel, OMP_begin_for,
%      OMP_clause_default, OMP_clause_ordered, OMP_clause_shared, 
%      OMP_clause_private, OMP_clause_firstprivate, OMP_clause_lastprivate, 
%      OMP_clause_reduction, OMP_clause_schedule
if ~strcmp(coder.target,'rtw'); return; end
coder.inline('always');
coder.allowpcode('plain');
if nargin~=nargout
error('All the input variables must also be output variables.');
end
coder.ceval('#pragma momp parallel for default', coder.opaque('const char *', 'none'));
coder.ceval('#{parfor');
