function OMP_begin_ordered %#codegen
%OMP_begin_ordered  specifies that the structured block following this
% directive is executed in the order in which iterations would be executed
% in a sequential loop.
%
%     OMP_begin_ordered
%
% It must be paired with OMP_end_ordered. This directive does not
% accept clauses. 
% 
% Description:
% An ordered directive must be within the dynamic extent of a for
% or parallel for construct. The for or parallel for directive to
% which the ordered construct binds must have an ordered clause.
% In the execution of a for or parallel for construct with an ordered
% clause, ordered constructs are executed strictly in the order in
% which they would be executed in a sequential execution of the loop.
%
% An iteration of a loop with a for construct must not execute the
% same ordered directive more than once, and it must not execute
% more than one ordered directive.
%
% See also OMP_end_ordered
if ~strcmp(coder.target,'rtw'); return; end
coder.inline('always')
coder.allowpcode('plain')
coder.ceval('#pragma momp ordered');
coder.ceval('#{ordered');
