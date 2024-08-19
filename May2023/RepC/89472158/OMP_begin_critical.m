function OMP_begin_critical(name) %#codegen
%OMP_begin_critical  specifies a region of code that must be executed by
%     only one thread at a time.
%
%     OMP_begin_critical()
%     OMP_begin_critical(name)
%
% It must be paired with OMP_end_critical. No implied barrier exists on 
% either entry to or exit from the master section. It does not accept clauses. 
% 
% Description:
% OMP_begin_critical() specifies a thread to waits at the
% beginning of a critical region until no other thread is executing 
% a critical region (anywhere in the program) with the same name.
%
% OMP_begin_critical(name) specifies the name of a critical section,
% where name is a constant character string. The optional name identifies 
% the critical region. Identifiers used to identify a critical region 
% have external linkage and are in a name space which is separate from 
% the name spaces used by labels, tags, members, and ordinary identifiers.
% All unnamed critical directives map to the same unspecified name.
%
% See also OMP_end_critical
if ~strcmp(coder.target,'rtw'); return; end
coder.inline('always')
coder.allowpcode('plain')
if nargin==0
coder.ceval('#pragma momp critical');
else
coder.ceval('#pragma momp critical', coder.opaque('void', name));
end
coder.ceval('#{critical');
