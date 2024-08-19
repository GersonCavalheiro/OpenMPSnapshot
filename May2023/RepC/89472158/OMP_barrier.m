function OMP_barrier %#codegen
%OMP_barrier  synchronizes all the threads in a team. 
%       OMP_barrier 
% When encountered, each thread in the team waits until all of 
% the others have reached this point. After all threads in the team 
% have encountered the barrier, each thread in the team begins 
% executing the statements after the barrier directive in parallel.
%
% This directive does not accept clauses.
if ~strcmp(coder.target,'rtw'); return; end
coder.inline('always')
coder.allowpcode('plain');
coder.ceval('#pragma momp barrier');
