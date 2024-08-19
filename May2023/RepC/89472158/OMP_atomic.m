function OMP_atomic %#codegen
%Specifies that a specific memory location must be updated atomically, rather
%than exposing it to the possibility of multiple, simultaneous writing threads. 
%
%     OMP_atomic()
%
% This directive does not accept clauses. No implied barrier exists on 
% either entry to or exit from the master section.
%
% Description:
% The expression statement must have one of the following forms:
%    x binop= expr, x++, ++x, x--, --x, where x is an lvalue expression 
% with scalar type, and expr is an expression with scalar type, and it 
% does not reference the object designated by x. binop is one of 
% +, *, -, /, &, ^, |, <<, or >>.
%
% Although it is implementation-defined whether an implementation replaces 
% all atomic directives with critical directives that have the same unique 
% name, the atomic directive permits better optimization. Often hardware 
% instructions are available that can perform the atomic update with 
% the least overhead. Only the load and store of the object designated 
% by x are atomic; the evaluation of expr is not atomic.
%
% All atomic references to the storage location x throughout the program 
% are required to have a compatible type.
%
% Note: This is an advanced function. It is advised that you check the C
% code to ensure the C code was generated correctly.
if ~strcmp(coder.target,'rtw'); return; end
coder.inline('always')
coder.allowpcode('plain')
coder.ceval('#pragma momp atomic');
