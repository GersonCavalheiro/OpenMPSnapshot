function OMP_flush(varargin) %#codegen
%OMP_flush  specifies that all threads have the same view of memory
%      for all shared objects.
%
%      OMP_flush(var1, var2, ...)
% The parameters list the variables that represent objects you want
% to synchronize. If var is not specified, all memory is flushed.
%
% This directive specifies a "cross-thread" sequence point at which the
% implementation is required to ensure that all threads in a team have
% a consistent view of certain objects (specified below) in memory.
% This means that previous evaluations of expressions that reference
% those objects are complete and subsequent evaluations have not yet begun.
% For example, compilers must restore the values of the objects from
% registers to memory, and hardware may need to flush write buffers
% to memory and reload the values of the objects from memory.
%
% If the objects that require synchronization can all be designated
% by variables, then those variables can be specified in the optional
% variable-list. If a pointer (an array) is present in the variable-list,
% the pointer itself is flushed, not the object the pointer refers to.
%
% A flush directive without a variable-list is implied for the following
% directives: OMP_barrier, OMP_begin/end_critical, OMP_begin/end_ordered,
% OMP_begin/end_parallel, OMP_end_sections, OMP_end_single, 
% OMP_begin/end_psections
%
% A flush is not implied if a nowait clause is present. It should be noted
% that the flush directive is not implied for any of the following:
% OMP_begin_sections, OMP_begin_single, OMP_begin/end_master,
%
% This directive does not accept clauses.
if ~strcmp(coder.target,'rtw'); return; end
coder.inline('always')
coder.allowpcode('plain')
coder.ceval('#pragma momp flush', varargin(:));
