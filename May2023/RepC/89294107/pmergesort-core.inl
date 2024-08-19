static inline void * _(ip)(void * key, void * lo, void * hi, int sense, context_t * ctx)
{
for (size_t lim = ELT_DIST(ctx, hi, lo); lim != 0; )
{
size_t half = lim >> 1;
void * mid = ELT_PTR_FWD(ctx, lo, half);
int result = CALL_CMP(ctx, key, mid);
if (result > sense)
{
lo = ELT_PTR_NEXT(ctx, mid);
lim = (lim - 1) >> 1;
}
else
{
lim = half;
}
}
return lo;
}
static inline void * _(ip_sym)(void * sym, void * lo, void * hi, context_t * ctx)
{
for (size_t lim = ELT_DIST(ctx, hi, lo); lim != 0; )
{
size_t half = lim >> 1;
void * mid = ELT_PTR_FWD(ctx, lo, half);
void * sym_mid = ELT_PTR_BCK(ctx, sym, half);
int result = CALL_CMP(ctx, mid, sym_mid);
if (result <= 0)
{
lo = ELT_PTR_NEXT(ctx, mid);
sym = ELT_PTR_PREV(ctx, sym_mid);
lim = (lim - 1) >> 1;
}
else
{
lim = half;
}
}
return lo;
}
static inline void _(inplace_merge_r2l)(void * lo, void * mi, void * hi, context_t * ctx)
{
size_t sz = ELT_SZ(ctx);
while (mi < hi)
{
size_t len = ELT_DIST(ctx, hi, lo);
void * ins;
size_t llen = ELT_DIST(ctx, mi, lo);
if (llen < _PMR_MIN_SUBMERGELEN2)
{
ins = mi;
while (lo < ins)
{
void * pins = ELT_PTR_PREV(ctx, ins);
if (CALL_CMP(ctx, pins, mi) > 0)
ins = pins;
else
break;
}
}
else
{
ins = _(ip)(mi, lo, mi, -1, ctx);
}
if (ins == mi)
break;
size_t rlen = len - llen;
void * nmi;
if (rlen < _PMR_MIN_SUBMERGELEN2)
{
nmi = ELT_PTR_NEXT(ctx, mi);
while (nmi < hi)
{
if (CALL_CMP(ctx, ins, nmi) > 0)
nmi = ELT_PTR_NEXT(ctx, nmi);
else
break;
}
}
else
{
nmi = _(ip)(ins, mi, hi, 0, ctx);
}
_M(rotate)(ins, mi, nmi, sz);
lo = ELT_PTR_FWD(ctx, ins, ELT_DIST(ctx, nmi, mi));
mi = nmi;
}
}
static inline void _(inplace_merge_l2r)(void * lo, void * mi, void * hi, context_t * ctx)
{
size_t sz = ELT_SZ(ctx);
while (lo < mi)
{
size_t len = ELT_DIST(ctx, hi, lo);
void * pmi = ELT_PTR_PREV(ctx, mi);
void * ins;
size_t rlen = ELT_DIST(ctx, hi, mi);
if (rlen < _PMR_MIN_SUBMERGELEN2)
{
ins = mi;
while (ins < hi)
{
if (CALL_CMP(ctx, ins, pmi) < 0)
ins = ELT_PTR_NEXT(ctx, ins);
else
break;
}
}
else
{
ins = _(ip)(pmi, mi, hi, 0, ctx);
}
if (ins == mi)
break;
void * pins = ELT_PTR_PREV(ctx, ins);
size_t llen = len - rlen;
if (llen < _PMR_MIN_SUBMERGELEN2)
{
while (lo < pmi)
{
void * ppmi = ELT_PTR_PREV(ctx, pmi);
if (CALL_CMP(ctx, pins, ppmi) < 0)
pmi = ppmi;
else
break;
}
}
else
{
pmi = _(ip)(pins, lo, pmi, -1, ctx);
}
_M(rotate)(pmi, mi, ins, sz);
hi = ELT_PTR_BCK(ctx, ins, ELT_DIST(ctx, mi, pmi));
mi = pmi;
}
}
static inline void _(inplace_merge)(void * lo, void * mi, void * hi, context_t * ctx, __unused aux_t * aux)
{
if (lo < mi && mi < hi)
{
if (mi - lo > hi - mi)
_(inplace_merge_r2l)(lo, mi, hi, ctx);
else
_(inplace_merge_l2r)(lo, mi, hi, ctx);
}
}
#if (PMR_PARALLEL_USE_GCD || PMR_PARALLEL_USE_PTHREADS) && _PMR_PARALLEL_MAY_SPAWN
static
#if PMR_PARALLEL_USE_PTHREADS
inline
#endif
void _(merge_spawn_pass)(void * arg)
{
pmergesort_pass_context_t * pass_ctx = arg;
aux_t * aux = &pass_ctx->auxes[0];
if (aux->rc == 0)
{
#if PMR_PARALLEL_USE_GCD && !_PMR_GCD_OVERCOMMIT
dispatch_semaphore_wait(pass_ctx->ctx->thpool->mutex, DISPATCH_TIME_FOREVER); 
#endif
aux_t laux;
laux.rc = 0;
laux.parent = aux;
laux.sz = 0;
laux.temp = NULL;
pass_ctx->effector(pass_ctx->lo, pass_ctx->mi, pass_ctx->hi, pass_ctx->ctx, &laux);
if (laux.rc != 0)
aux->rc = laux.rc; 
_aux_free(&laux);
#if PMR_PARALLEL_USE_GCD && !_PMR_GCD_OVERCOMMIT
dispatch_semaphore_signal(pass_ctx->ctx->thpool->mutex); 
#endif
}
PMR_FREE(arg); 
}
#if PMR_PARALLEL_USE_PTHREADS
static void * _(merge_spawn_pass_ex)(void * arg)
{
_(merge_spawn_pass)(arg);
return NULL;
}
#endif
#endif 
static void _(inplace_symmerge)(void * lo, void * mi, void * hi, context_t * ctx, aux_t * aux)
{
size_t sz = ELT_SZ(ctx);
while (lo < mi && mi < hi)
{
size_t llen = ELT_DIST(ctx, mi, lo);
if (llen < _PMR_MIN_SUBMERGELEN1)
{
_(inplace_merge_l2r)(lo, mi, hi, ctx);
break; 
}
size_t len = ELT_DIST(ctx, hi, lo);
size_t rlen = len - llen;
if (rlen < _PMR_MIN_SUBMERGELEN1)
{
_(inplace_merge_r2l)(lo, mi, hi, ctx);
break; 
}
void * mid = ELT_PTR_FWD(ctx, lo, len >> 1);
void * bound = ELT_PTR_FWD(ctx, mid, llen);
void * start;
if (mid < mi) 
{
void * lbound = ELT_PTR_BCK(ctx, bound, len);
start = _(ip_sym)(ELT_PTR_PREV(ctx, hi), lbound, ELT_PTR_FWD(ctx, lbound, rlen), ctx);
}
else 
{
start = _(ip_sym)(ELT_PTR_PREV(ctx, bound), lo, mi, ctx);
}
void * end = ELT_PTR_FWD(ctx, lo, ELT_DIST(ctx, bound, start));
_M(rotate)(start, mi, end, sz);
if (lo < start && start < mid)
{
#if _PMR_PARALLEL_MAY_SPAWN
#if PMR_PARALLEL_USE_GCD || PMR_PARALLEL_USE_PTHREADS
if (ctx->thpool != NULL && len > ctx->cut_off)
{
pmergesort_pass_context_t * pass_ctx = PMR_MALLOC(sizeof(pmergesort_pass_context_t));
pass_ctx->ctx = ctx;
pass_ctx->bsz = 0;
pass_ctx->dbl_bsz = 0;
pass_ctx->chunksz = 0;
pass_ctx->numchunks = 0;
pass_ctx->lo = lo;
pass_ctx->mi = start;
pass_ctx->hi = mid;
pass_ctx->effector = ctx->merge_effector;
pass_ctx->auxes = aux->parent;
#if PMR_PARALLEL_USE_PTHREADS
thr_pool_queue(ctx->thpool, _(merge_spawn_pass_ex), pass_ctx);
#elif PMR_PARALLEL_USE_GCD
dispatch_group_async_f(ctx->thpool->group, ctx->thpool->queue, pass_ctx, _(merge_spawn_pass));
#endif
}
else
_(inplace_symmerge)(lo, start, mid, ctx, aux);
#elif PMR_PARALLEL_USE_OMP
#pragma omp task if (len > ctx->cut_off) default(none) firstprivate(lo, start, mid, ctx)
_(inplace_symmerge)(lo, start, mid, ctx, NULL);
#endif 
#else
_(inplace_symmerge)(lo, start, mid, ctx, NULL);
#endif 
}
lo = mid;
mi = end;
}
}
static __attribute__((unused)) void _(aux_symmerge)(void * lo, void * mi, void * hi, context_t * ctx, aux_t * aux)
{
#if PMR_PARALLEL_USE_OMP
aux_t * paux = aux->parent;
#endif
size_t sz = ELT_SZ(ctx);
while (lo < mi && mi < hi)
{
size_t llen = ELT_DIST(ctx, mi, lo);
if (llen < _PMR_MIN_SUBMERGELEN1)
{
_(inplace_merge_l2r)(lo, mi, hi, ctx);
break; 
}
size_t len = ELT_DIST(ctx, hi, lo);
size_t rlen = len - llen;
if (rlen < _PMR_MIN_SUBMERGELEN1)
{
_(inplace_merge_r2l)(lo, mi, hi, ctx);
break; 
}
void * mid = ELT_PTR_FWD(ctx, lo, len >> 1);
void * bound = ELT_PTR_FWD(ctx, mid, llen);
void * start;
if (mid < mi) 
{
void * lbound = ELT_PTR_BCK(ctx, bound, len);
start = _(ip_sym)(ELT_PTR_PREV(ctx, hi), lbound, ELT_PTR_FWD(ctx, lbound, rlen), ctx);
}
else 
{
start = _(ip_sym)(ELT_PTR_PREV(ctx, bound), lo, mi, ctx);
}
void * end = ELT_PTR_FWD(ctx, lo, ELT_DIST(ctx, bound, start));
_M(rotate_aux)(start, mi, end, sz, aux);
if (aux->rc != 0)
break; 
if (lo < start && start < mid)
{
#if _PMR_PARALLEL_MAY_SPAWN
#if PMR_PARALLEL_USE_GCD || PMR_PARALLEL_USE_PTHREADS
if (ctx->thpool != NULL && len > ctx->cut_off)
{
pmergesort_pass_context_t * pass_ctx = PMR_MALLOC(sizeof(pmergesort_pass_context_t));
pass_ctx->ctx = ctx;
pass_ctx->bsz = 0;
pass_ctx->dbl_bsz = 0;
pass_ctx->chunksz = 0;
pass_ctx->numchunks = 0;
pass_ctx->lo = lo;
pass_ctx->mi = start;
pass_ctx->hi = mid;
pass_ctx->effector = ctx->merge_effector;
pass_ctx->auxes = aux->parent;
#if PMR_PARALLEL_USE_PTHREADS
thr_pool_queue(ctx->thpool, _(merge_spawn_pass_ex), pass_ctx);
#elif PMR_PARALLEL_USE_GCD
dispatch_group_async_f(ctx->thpool->group, ctx->thpool->queue, pass_ctx, _(merge_spawn_pass));
#endif
}
else
{
_(aux_symmerge)(lo, start, mid, ctx, aux);
if (aux->rc != 0)
break; 
}
#elif PMR_PARALLEL_USE_OMP
#pragma omp task if (len > ctx->cut_off) default(none) firstprivate(lo, start, mid, ctx, paux)
if (paux->rc == 0)
{
aux_t laux;
laux.rc = 0;
laux.parent = paux;
laux.sz = 0;
laux.temp = NULL;
_(aux_symmerge)(lo, start, mid, ctx, &laux);
if (laux.rc != 0)
paux->rc = laux.rc; 
_aux_free(&laux);
}
#endif 
#else
_(aux_symmerge)(lo, start, mid, ctx, aux);
if (aux->rc != 0)
break; 
#endif 
}
lo = mid;
mi = end;
}
}
static inline void _(aux_merge_r)(void * lo, void * mi, void * hi, context_t * ctx, aux_t * aux)
{
size_t rsz = ELT_DIST(ctx, hi, mi);
if (rsz < _PMR_MIN_SUBMERGELEN1)
{
_(inplace_merge_r2l)(lo, mi, hi, ctx);
return; 
}
size_t sz = ELT_SZ(ctx);
void * tmp = _aux_alloc(aux, ELT_OF_SZ(rsz, sz));
if (tmp == NULL)
return; 
_M(copy)(mi, tmp, rsz, sz);
void * dst = hi;
void * lsrclo = lo;
void * lsrchi = ELT_PTR_PREV(ctx, mi);
void * rsrclo = tmp;
void * rsrchi = ELT_PTR_FWD(ctx, tmp, rsz - 1);
while (lsrchi >= lsrclo && rsrchi >= rsrclo)
{
dst = ELT_PTR_PREV(ctx, dst);
int rc = CALL_CMP(ctx, lsrchi, rsrchi);
if (rc > 0)
{
_M(copy)(lsrchi, dst, 1, sz);
lsrchi = ELT_PTR_PREV(ctx, lsrchi);
}
else
{
_M(copy)(rsrchi, dst, 1, sz);
rsrchi = ELT_PTR_PREV(ctx, rsrchi);
}
}
if (rsrchi >= rsrclo)
{
size_t tailsz = ELT_DIST(ctx, rsrchi, rsrclo) + 1;
_M(copy)(rsrclo, ELT_PTR_BCK(ctx, dst, tailsz), tailsz, sz);
}
}
static inline void _(aux_merge_l)(void * lo, void * mi, void * hi, context_t * ctx, aux_t * aux)
{
size_t lsz = ELT_DIST(ctx, mi, lo);
if (lsz < _PMR_MIN_SUBMERGELEN1)
{
_(inplace_merge_l2r)(lo, mi, hi, ctx);
return; 
}
size_t sz = ELT_SZ(ctx);
void * tmp = _aux_alloc(aux, ELT_OF_SZ(lsz, sz));
if (tmp == NULL)
return; 
_M(copy)(lo, tmp, lsz, sz);
void * dst = lo;
void * lsrclo = tmp;
void * lsrchi = ELT_PTR_FWD(ctx, tmp, lsz);
void * rsrclo = mi;
void * rsrchi = hi;
while (lsrclo < lsrchi && rsrclo < rsrchi)
{
int rc = CALL_CMP(ctx, lsrclo, rsrclo);
if (rc <= 0)
{
_M(copy)(lsrclo, dst, 1, sz);
lsrclo = ELT_PTR_NEXT(ctx, lsrclo);
}
else
{
_M(copy)(rsrclo, dst, 1, sz);
rsrclo = ELT_PTR_NEXT(ctx, rsrclo);
}
dst = ELT_PTR_NEXT(ctx, dst);
}
if (lsrclo < lsrchi)
{
_M(copy)(lsrclo, dst, ELT_DIST(ctx, lsrchi, lsrclo), sz);
}
}
static inline void _(aux_merge)(void * lo, void * mi, void * hi, context_t * ctx, aux_t * aux)
{
if (lo < mi && mi < hi)
{
void * inslo = _(ip)(mi, lo, mi, -1, ctx);
if (inslo == mi)
return; 
void * inshi = _(ip)(ELT_PTR_PREV(ctx, mi), mi, hi, 0, ctx);
#if 1
if (CALL_CMP(ctx, inslo, ELT_PTR_PREV(ctx, inshi)) > 0)
{
_M(rotate_aux)(inslo, mi, inshi, ELT_SZ(ctx), aux);
return; 
}
#endif
if (mi - inslo > inshi - mi)
_(aux_merge_r)(inslo, mi, inshi, ctx, aux);
else
_(aux_merge_l)(inslo, mi, inshi, ctx, aux);
}
}
static inline void * _(next_run)(void * lo, void * hi, context_t * ctx)
{
if (lo < hi)
{
void * cur = ELT_PTR_NEXT(ctx, lo);
if (cur >= hi)
return lo;
int r = ISIGN(CALL_CMP(ctx, lo, cur));
while (cur < hi)
{
void * ncur = ELT_PTR_NEXT(ctx, cur);
if (ncur >= hi)
break;
int rc = ISIGN(CALL_CMP(ctx, cur, ncur));
if (r == rc)
{
cur = ncur;
}
else
{
if (rc == 0)
{
if (r > 0) 
break;
cur = ncur;
}
else if (r == 0)
{
if (rc > 0) 
break;
cur = ncur;
r = rc; 
}
else
break; 
}
}
if (r > 0)
{
_M(reverse)(lo, cur, ELT_SZ(ctx));
}
return cur;
}
else
return lo;
}
static inline void _(binsort)(void * lo, void * mi, void * hi, context_t * ctx, __unused aux_t * aux)
{
size_t sz = ELT_SZ(ctx);
mi = ELT_PTR_NEXT(ctx, mi); 
size_t llen = ELT_DIST(ctx, mi, lo); 
while (mi < hi)
{
void * ins;
if (llen < _PMR_MIN_SUBMERGELEN2)
{
ins = mi;
while (lo < ins)
{
void * pins = ELT_PTR_PREV(ctx, ins);
if (CALL_CMP(ctx, pins, mi) > 0)
ins = pins;
else
break;
}
}
else
{
ins = _(ip)(mi, lo, mi, -1, ctx);
}
if (ins < mi)
{
uint8_t t[ELT_OF_SZ(1, sz)];
_M(copy)(mi, t, 1, sz);
_M(move_right)(ins, ELT_DIST(ctx, mi, ins), 1, sz);
_M(copy)(t, ins, 1, sz);
}
mi = ELT_PTR_NEXT(ctx, mi);
llen++;
}
}
static inline void _(binsort_run)(void * lo, __unused void * mi, void * hi, context_t * ctx, __unused aux_t * aux)
{
void * end = _(next_run)(lo, hi, ctx);
if (end < hi)
_(binsort)(lo, end, hi, ctx, NULL);
}
static inline void _(binsort_mergerun)(void * lo, __unused void * mi, void * hi, context_t * ctx, __unused aux_t * aux)
{
void * end = ELT_PTR_NEXT(ctx, _(next_run)(lo, hi, ctx));
while (end < hi)
{
void * end0 = ELT_PTR_NEXT(ctx, _(next_run)(end, hi, ctx));
if (end0 > hi)
end0 = hi;
_(inplace_merge)(lo, end, end0, ctx, NULL);
end = end0;
}
}
#if PMR_PARALLEL_USE_GCD || PMR_PARALLEL_USE_PTHREADS
static
#if PMR_PARALLEL_USE_PTHREADS
inline
#endif
void _(sort_chunk_pass)(void * arg, size_t chunk)
{
pmergesort_pass_context_t * pass_ctx = arg;
aux_t * aux = &pass_ctx->auxes[chunk];
int last = (chunk < pass_ctx->numchunks - 1) ? 0 : 1;
void * a = ELT_PTR_FWD(pass_ctx->ctx, pass_ctx->lo, pass_ctx->chunksz * chunk);
void * b = ELT_PTR_FWD(pass_ctx->ctx, a, pass_ctx->bsz);
void * c = last == 0 ? ELT_PTR_FWD(pass_ctx->ctx, a, pass_ctx->chunksz) : pass_ctx->hi;
while (b <= c)
{
pass_ctx->effector(a, a, b, pass_ctx->ctx, aux);
if (aux->rc != 0)
break;
a = b;
b = ELT_PTR_FWD(pass_ctx->ctx, b, pass_ctx->bsz);
}
if (last != 0 && aux->rc == 0)
pass_ctx->effector(a, a, c, pass_ctx->ctx, aux);
}
static
#if PMR_PARALLEL_USE_PTHREADS
inline
#endif
void _(merge_chunks_pass)(void * arg, size_t chunk)
{
pmergesort_pass_context_t * pass_ctx = arg;
#if PMR_PARALLEL_USE_GCD && _PMR_PARALLEL_MAY_SPAWN && !_PMR_GCD_OVERCOMMIT
dispatch_semaphore_wait(pass_ctx->ctx->thpool->mutex, DISPATCH_TIME_FOREVER); 
#endif
aux_t * aux = &pass_ctx->auxes[chunk];
int last = (chunk < pass_ctx->numchunks - 1) ? 0 : 1;
void * a = ELT_PTR_FWD(pass_ctx->ctx, pass_ctx->lo, pass_ctx->chunksz * chunk);
void * b = ELT_PTR_FWD(pass_ctx->ctx, a, pass_ctx->dbl_bsz);
void * c = last == 0 ? ELT_PTR_FWD(pass_ctx->ctx, a, pass_ctx->chunksz) : pass_ctx->hi;
while (b <= c)
{
pass_ctx->effector(a, ELT_PTR_FWD(pass_ctx->ctx, a, pass_ctx->bsz), b, pass_ctx->ctx, aux);
if (aux->rc != 0)
break;
a = b;
b = ELT_PTR_FWD(pass_ctx->ctx, b, pass_ctx->dbl_bsz);
}
if (last != 0 && aux->rc == 0)
pass_ctx->effector(a, ELT_PTR_FWD(pass_ctx->ctx, a, pass_ctx->bsz), c, pass_ctx->ctx, aux);
#if PMR_PARALLEL_USE_GCD && _PMR_PARALLEL_MAY_SPAWN && !_PMR_GCD_OVERCOMMIT
dispatch_semaphore_signal(pass_ctx->ctx->thpool->mutex); 
#endif
}
#if PMR_PARALLEL_USE_PTHREADS
static void * _(sort_chunk_pass_ex)(void * arg)
{
_(sort_chunk_pass)(arg, ((pmergesort_pass_context_t *)arg)->chunk);
return NULL;
}
static void * _(merge_chunks_pass_ex)(void * arg)
{
_(merge_chunks_pass)(arg, ((pmergesort_pass_context_t *)arg)->chunk);
return NULL;
}
static inline int _(pmergesort_impl)(context_t * ctx)
{
aux_t auxes[ctx->ncpu];
for (int i = 0; i < ctx->ncpu; i++)
auxes[i] = (aux_t){ .parent = &auxes[i] };
void * lo = (void *)ctx->base;
void * hi = ELT_PTR_FWD(ctx, lo, ctx->n);
size_t bsz = ctx->bsize;
size_t npercpu = ctx->npercpu;
{
size_t chunksz = IDIV_UP(npercpu, bsz) * bsz;
size_t numchunks = IDIV_UP(ctx->n, chunksz);
pmergesort_pass_context_t pass1_ctx_base;
pass1_ctx_base.ctx = ctx;
pass1_ctx_base.bsz = bsz;
pass1_ctx_base.chunksz = chunksz;
pass1_ctx_base.numchunks = numchunks;
pass1_ctx_base.lo = lo;
pass1_ctx_base.mi = NULL;
pass1_ctx_base.hi = hi;
pass1_ctx_base.effector = ctx->sort_effector;
pass1_ctx_base.auxes = auxes;
if (numchunks > 1)
{
pmergesort_pass_context_t pass1_ctx[numchunks];
for (size_t chunk = 0; chunk < numchunks; chunk++)
{
pass1_ctx[chunk] = pass1_ctx_base;
pass1_ctx[chunk].chunk = chunk;
thr_pool_queue(ctx->thpool, _(sort_chunk_pass_ex), &pass1_ctx[chunk]);
}
thr_pool_wait(ctx->thpool);
for (int i = 0; i < numchunks; i++)
{
if (auxes[i].rc != 0)
goto bail_out;
}
}
else
{
pass1_ctx_base.chunk = 0;
_(sort_chunk_pass_ex)(&pass1_ctx_base);
goto bail_out;
}
}
pmergesort_pass_context_t pass2_ctx_base;
pass2_ctx_base.ctx = ctx;
pass2_ctx_base.lo = lo;
pass2_ctx_base.mi = NULL;
pass2_ctx_base.hi = hi;
pass2_ctx_base.effector = ctx->merge_effector;
pass2_ctx_base.auxes = auxes;
while (bsz < ctx->n)
{
size_t dbl_bsz = bsz << 1;
size_t chunksz = IDIV_UP(npercpu, dbl_bsz) * dbl_bsz;
size_t numchunks = IDIV_UP(ctx->n, chunksz);
pass2_ctx_base.bsz = bsz;
pass2_ctx_base.dbl_bsz = dbl_bsz;
pass2_ctx_base.chunksz = chunksz;
pass2_ctx_base.numchunks = numchunks;
if (numchunks > 1)
{
for (int i = (int)numchunks; i < ctx->ncpu; i++)
_aux_free(&auxes[i]);
pmergesort_pass_context_t pass2_ctx[numchunks];
for (size_t chunk = 0; chunk < numchunks; chunk++)
{
pass2_ctx[chunk] = pass2_ctx_base;
pass2_ctx[chunk].chunk = chunk;
thr_pool_queue(ctx->thpool, _(merge_chunks_pass_ex), &pass2_ctx[chunk]);
}
thr_pool_wait(ctx->thpool);
for (int i = 0; i < numchunks; i++)
{
if (auxes[i].rc != 0)
goto bail_out;
}
}
else
{
pass2_ctx_base.chunk = 0;
_(merge_chunks_pass_ex)(&pass2_ctx_base);
#if _PMR_PARALLEL_MAY_SPAWN
thr_pool_wait(ctx->thpool);
#endif
}
bsz = dbl_bsz;
}
bail_out:;
int rc = 0;
for (int i = 0; i < ctx->ncpu; i++)
{
_aux_free(&auxes[i]);
if (rc == 0)
rc = auxes[i].rc;
}
return rc;
}
#elif PMR_PARALLEL_USE_GCD
static inline int _(pmergesort_impl)(context_t * ctx)
{
dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, _PMR_DISPATCH_QUEUE_FLAGS);
thr_pool_t pool;
pool.queue = queue;
#if _PMR_PARALLEL_MAY_SPAWN
pool.group = NULL;
#if !_PMR_GCD_OVERCOMMIT
pool.mutex = NULL;
#endif
#endif
ctx->thpool = &pool;
aux_t auxes[ctx->ncpu];
for (int i = 0; i < ctx->ncpu; i++)
auxes[i] = (aux_t){ .parent = &auxes[i] };
void * lo = (void *)ctx->base;
void * hi = ELT_PTR_FWD(ctx, lo, ctx->n);
size_t bsz = ctx->bsize;
size_t npercpu = ctx->npercpu;
{
size_t chunksz = IDIV_UP(npercpu, bsz) * bsz;
size_t numchunks = IDIV_UP(ctx->n, chunksz);
pmergesort_pass_context_t pass1_ctx;
pass1_ctx.ctx = ctx;
pass1_ctx.bsz = bsz;
pass1_ctx.chunksz = chunksz;
pass1_ctx.numchunks = numchunks;
pass1_ctx.lo = lo;
pass1_ctx.mi = NULL;
pass1_ctx.hi = hi;
pass1_ctx.effector = ctx->sort_effector;
pass1_ctx.auxes = auxes;
if (numchunks > 1)
{
dispatch_apply_f(numchunks, queue, &pass1_ctx, _(sort_chunk_pass));
for (int i = 0; i < numchunks; i++)
{
if (auxes[i].rc != 0)
goto bail_out;
}
}
else
{
_(sort_chunk_pass)(&pass1_ctx, 0);
goto bail_out;
}
}
#if _PMR_PARALLEL_MAY_SPAWN
pool.group = dispatch_group_create();
#if !_PMR_GCD_OVERCOMMIT
pool.mutex = dispatch_semaphore_create(ctx->ncpu);
#endif
#endif
{
pmergesort_pass_context_t pass2_ctx;
pass2_ctx.ctx = ctx;
pass2_ctx.lo = lo;
pass2_ctx.mi = NULL;
pass2_ctx.hi = hi;
pass2_ctx.effector = ctx->merge_effector;
pass2_ctx.auxes = auxes;
while (bsz < ctx->n)
{
size_t dbl_bsz = bsz << 1;
size_t chunksz = IDIV_UP(npercpu, dbl_bsz) * dbl_bsz;
size_t numchunks = IDIV_UP(ctx->n, chunksz);
pass2_ctx.bsz = bsz;
pass2_ctx.dbl_bsz = dbl_bsz;
pass2_ctx.chunksz = chunksz;
pass2_ctx.numchunks = numchunks;
if (numchunks > 1)
{
for (int i = (int)numchunks; i < ctx->ncpu; i++)
_aux_free(&auxes[i]);
dispatch_apply_f(numchunks, queue, &pass2_ctx, _(merge_chunks_pass));
#if _PMR_PARALLEL_MAY_SPAWN
dispatch_group_wait(pool.group, DISPATCH_TIME_FOREVER);
#endif
for (int i = 0; i < numchunks; i++)
{
if (auxes[i].rc != 0)
goto bail_out;
}
}
else
{
_(merge_chunks_pass)(&pass2_ctx, 0);
#if _PMR_PARALLEL_MAY_SPAWN
dispatch_group_wait(pool.group, DISPATCH_TIME_FOREVER);
#endif
}
bsz = dbl_bsz;
}
}
bail_out:;
#if _PMR_PARALLEL_MAY_SPAWN
if (pool.group != NULL)
dispatch_release(DISPATCH_OBJECT_T(pool.group));
#if !_PMR_GCD_OVERCOMMIT
if (pool.mutex != NULL)
dispatch_release(DISPATCH_OBJECT_T(pool.mutex));
#endif
#endif
int rc = 0;
for (int i = 0; i < ctx->ncpu; i++)
{
_aux_free(&auxes[i]);
if (rc == 0)
rc = auxes[i].rc;
}
return rc;
}
#endif
#elif PMR_PARALLEL_USE_OMP
static inline int _(pmergesort_impl)(context_t * ctx)
{
aux_t auxes[ctx->ncpu];
for (int i = 0; i < ctx->ncpu; i++)
auxes[i] = (aux_t){ .parent = &auxes[i] };
void * lo = (void *)ctx->base;
void * hi = ELT_PTR_FWD(ctx, lo, ctx->n);
size_t bsz = ctx->bsize;
size_t npercpu = ctx->npercpu;
{
size_t chunksz = IDIV_UP(npercpu, bsz) * bsz;
size_t numchunks = IDIV_UP(ctx->n, chunksz);
#pragma omp parallel num_threads(numchunks)
#pragma omp for
for (size_t chunk = 0; chunk < numchunks; chunk++)
{
aux_t * aux = &auxes[chunk];
int last = (chunk < numchunks - 1) ? 0 : 1;
void * a = ELT_PTR_FWD(ctx, lo, chunksz * chunk);
void * b = ELT_PTR_FWD(ctx, a, bsz);
void * c = last == 0 ? ELT_PTR_FWD(ctx, a, chunksz) : hi;
while (b <= c)
{
ctx->sort_effector(a, a, b, ctx, aux);
if (aux->rc != 0)
break;
a = b;
b = ELT_PTR_FWD(ctx, b, bsz);
}
if (last != 0 && aux->rc == 0)
ctx->sort_effector(a, a, c, ctx, aux);
}
for (int i = 0; i < numchunks; i++)
{
if (auxes[i].rc != 0)
goto bail_out;
}
}
{
__unused size_t ncpu = ctx->thpool != NULL ? ctx->thpool->ncpu : 0;
while (bsz < ctx->n)
{
size_t dbl_bsz = bsz << 1;
size_t chunksz = IDIV_UP(npercpu, dbl_bsz) * dbl_bsz;
size_t numchunks = IDIV_UP(ctx->n, chunksz);
#pragma omp parallel num_threads(ncpu != 0 ? ncpu : numchunks)
#pragma omp for
for (size_t chunk = 0; chunk < numchunks; chunk++)
{
aux_t * aux = &auxes[chunk];
int last = (chunk < numchunks - 1) ? 0 : 1;
void * a = ELT_PTR_FWD(ctx, lo, chunksz * chunk);
void * b = ELT_PTR_FWD(ctx, a, dbl_bsz);
void * c = last == 0 ? ELT_PTR_FWD(ctx, a, chunksz) : hi;
while (b <= c)
{
ctx->merge_effector(a, ELT_PTR_FWD(ctx, a, bsz), b, ctx, aux);
if (aux->rc != 0)
break;
a = b;
b = ELT_PTR_FWD(ctx, b, dbl_bsz);
}
if (last != 0 && aux->rc == 0)
ctx->merge_effector(a, ELT_PTR_FWD(ctx, a, bsz), c, ctx, aux);
}
#pragma omp taskwait
for (int i = 0; i < numchunks; i++)
{
if (auxes[i].rc != 0)
goto bail_out;
}
bsz = dbl_bsz;
}
}
bail_out:;
int rc = 0;
for (int i = 0; i < ctx->ncpu; i++)
{
_aux_free(&auxes[i]);
if (rc == 0)
rc = auxes[i].rc;
}
return rc;
}
#endif 
static inline void _(symmergesort)(context_t * ctx)
{
if (ctx->n < _PMR_BLOCKLEN_MTHRESHOLD0 * _PMR_BLOCKLEN_SYMMERGE)
{
void * lo = (void *)ctx->base;
void * hi = ELT_PTR_FWD(ctx, lo, ctx->n);
_(_PMR_PRESORT)(lo, lo, hi, ctx, NULL);
return;
}
#if PMR_PARALLEL_USE_GCD || PMR_PARALLEL_USE_PTHREADS || PMR_PARALLEL_USE_OMP
for (int ncpu = ctx->ncpu; ncpu > 1; ncpu--)
{
size_t npercpu = IDIV_UP(ctx->n, ncpu);
if (npercpu >= _PMR_BLOCKLEN_MTHRESHOLD * _PMR_BLOCKLEN_SYMMERGE)
{
ctx->npercpu = npercpu;
ctx->bsize = _PMR_BLOCKLEN_SYMMERGE;
ctx->sort_effector = _(_PMR_PRESORT);
ctx->merge_effector = _(inplace_symmerge);
#if PMR_PARALLEL_USE_OMP && _PMR_PARALLEL_MAY_SPAWN
thr_pool_t pool;
pool.ncpu = ctx->ncpu; 
ctx->thpool = &pool;
#endif
(void)_(pmergesort_impl)(ctx);
return;
}
}
#endif
#if (PMR_PARALLEL_USE_GCD || PMR_PARALLEL_USE_PTHREADS) && _PMR_PARALLEL_MAY_SPAWN
ctx->thpool = NULL; 
#endif
void * lo = (void *)ctx->base;
void * hi = ELT_PTR_FWD(ctx, lo, ctx->n);
size_t bsz = _PMR_BLOCKLEN_SYMMERGE;
void * a = lo;
void * b = ELT_PTR_FWD(ctx, a, bsz);
while (b <= hi)
{
_(_PMR_PRESORT)(a, a, b, ctx, NULL);
a = b;
b = ELT_PTR_FWD(ctx, b, bsz);
}
_(_PMR_PRESORT)(a, a, hi, ctx, NULL);
while (bsz < ctx->n)
{
size_t bsz1 = bsz << 1;
a = lo;
b = ELT_PTR_FWD(ctx, a, bsz1);
while (b <= hi)
{
_(inplace_symmerge)(a, ELT_PTR_FWD(ctx, a, bsz), b, ctx, NULL);
a = b;
b = ELT_PTR_FWD(ctx, b, bsz1);
}
_(inplace_symmerge)(a, ELT_PTR_FWD(ctx, a, bsz), hi, ctx, NULL);
bsz = bsz1;
}
}
static inline int _(pmergesort)(context_t * ctx)
{
if (ctx->n < _PMR_BLOCKLEN_MTHRESHOLD0 * _PMR_BLOCKLEN_SYMMERGE)
{
void * lo = (void *)ctx->base;
void * hi = ELT_PTR_FWD(ctx, lo, ctx->n);
_(_PMR_PRESORT)(lo, lo, hi, ctx, NULL);
return 0;
}
#if PMR_PARALLEL_USE_GCD || PMR_PARALLEL_USE_PTHREADS || PMR_PARALLEL_USE_OMP
for (int ncpu = ctx->ncpu; ncpu > 1; ncpu--)
{
size_t npercpu = IDIV_UP(ctx->n, ncpu);
if (npercpu >= _PMR_BLOCKLEN_MTHRESHOLD * _PMR_BLOCKLEN_MERGE)
{
ctx->npercpu = npercpu;
ctx->bsize = _PMR_BLOCKLEN_MERGE;
ctx->sort_effector = _(_PMR_PRESORT);
ctx->merge_effector = _(aux_merge);
return _(pmergesort_impl)(ctx);
}
}
#endif
#if (PMR_PARALLEL_USE_GCD || PMR_PARALLEL_USE_PTHREADS) && _PMR_PARALLEL_MAY_SPAWN
ctx->thpool = NULL; 
#endif
void * lo = (void *)ctx->base;
void * hi = ELT_PTR_FWD(ctx, lo, ctx->n);
size_t bsz = _PMR_BLOCKLEN_MERGE;
void * a = lo;
void * b = ELT_PTR_FWD(ctx, a, bsz);
while (b <= hi)
{
_(_PMR_PRESORT)(a, a, b, ctx, NULL);
a = b;
b = ELT_PTR_FWD(ctx, b, bsz);
}
_(_PMR_PRESORT)(a, a, hi, ctx, NULL);
aux_t aux;
memset(&aux, 0, sizeof(aux));
while (bsz < ctx->n)
{
size_t bsz1 = bsz << 1;
a = lo;
b = ELT_PTR_FWD(ctx, a, bsz1);
while (b <= hi)
{
_(aux_merge)(a, ELT_PTR_FWD(ctx, a, bsz), b, ctx, &aux);
if (aux.rc != 0)
goto bail_out;
a = b;
b = ELT_PTR_FWD(ctx, b, bsz1);
}
_(aux_merge)(a, ELT_PTR_FWD(ctx, a, bsz), hi, ctx, &aux);
if (aux.rc != 0)
goto bail_out;
bsz = bsz1;
}
bail_out:;
_aux_free(&aux);
return aux.rc;
}
static __attribute__((unused)) void _(wrap_sort)(void * lo, __unused void * mi, void * hi, context_t * ctx, aux_t * aux)
{
aux->rc = CALL_SORT(ctx, lo, ELT_DIST(ctx, hi, lo)); 
}
static inline int _(wrapmergesort)(context_t * ctx)
{
#if PMR_PARALLEL_USE_GCD || PMR_PARALLEL_USE_PTHREADS || PMR_PARALLEL_USE_OMP
if (ctx->n >= 2 * _PMR_BLOCKLEN_MTHRESHOLD0 * _PMR_BLOCKLEN_SYMMERGE)
{
for (int ncpu = ctx->ncpu; ncpu > 1; ncpu--)
{
size_t npercpu = IDIV_UP(ctx->n, ncpu);
if (npercpu >= 2 * _PMR_BLOCKLEN_MTHRESHOLD * _PMR_BLOCKLEN_SYMMERGE)
{
ctx->npercpu = npercpu;
ctx->bsize = npercpu;
ctx->sort_effector = _(wrap_sort);
ctx->merge_effector = _(aux_symmerge);
#if PMR_PARALLEL_USE_OMP && _PMR_PARALLEL_MAY_SPAWN
thr_pool_t pool;
pool.ncpu = ctx->ncpu; 
ctx->thpool = &pool;
#endif
return _(pmergesort_impl)(ctx);
}
}
}
#endif
return CALL_SORT(ctx, (void *)ctx->base, ctx->n);
}
#if _PMR_CORE_PROFILE
static inline void _(insertionsort)(context_t * ctx)
{
void * lo = (void *)ctx->base;
void * hi = ELT_PTR_FWD(ctx, lo, ctx->n);
_(binsort)(lo, lo, hi, ctx, NULL);
}
#endif
#if _PMR_CORE_PROFILE
static inline void _(insertionsort_run)(context_t * ctx)
{
void * lo = (void *)ctx->base;
void * hi = ELT_PTR_FWD(ctx, lo, ctx->n);
_(binsort_run)(lo, lo, hi, ctx, NULL);
}
#endif
#if _PMR_CORE_PROFILE
static inline void _(insertionsort_mergerun)(context_t * ctx)
{
void * lo = (void *)ctx->base;
void * hi = ELT_PTR_FWD(ctx, lo, ctx->n);
_(binsort_mergerun)(lo, lo, hi, ctx, NULL);
}
#endif
