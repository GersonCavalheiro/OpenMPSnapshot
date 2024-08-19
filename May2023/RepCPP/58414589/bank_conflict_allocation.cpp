

#include "gpu/jit/codegen/bank_conflict_allocation.hpp"

#include "common/verbose.hpp"
#include "gpu/jit/ir/fma.hpp"
#include "gpu/jit/utils/utils.hpp"

#include <sstream>
#include <string>
#include <vector>
#include <initializer_list>

#if defined(__GNUC__) && __GNUC__ == 7
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

namespace {

struct hw_context_t {
hw_context_t(ngen::HW hw, int regs)
: hw(hw), regs(regs), reg_size(ngen::GRF::bytes(hw)) {
int bank0 = reg_bank(0);
for (int i = 1; i < regs; i++)
if (reg_bank(i) != bank0) {
reg_bank_stride = i;
break;
}
ir_assert(reg_bank_stride != -1);

bank_masks.resize(ngen::Bundle::bank_count(hw));
bundle_masks.resize(ngen::Bundle::bundle_count(hw));

for (int i = 0; i < regs; i++) {
int bank = reg_bank(i);
int bundle = reg_bundle(i);
if (i < 64) {
bank_masks[bank] |= (1ull << i);
bundle_masks[bundle] |= (1ull << i);
} else {
int j = (i % 64);
ir_assert((bank_masks[bank] & (1ull << j)) != 0);
if (hw > ngen::HW::XeLP)
ir_assert((bundle_masks[bundle] & (1ull << j)) != 0);
}
}
}

int reg_bank(int reg) const {
auto bundle = ngen::Bundle::locate(hw, ngen::GRF(reg));
return bundle.bank_id;
}

int reg_bundle(int reg) const {
auto bundle = ngen::Bundle::locate(hw, ngen::GRF(reg));
return bundle.bundle_id;
}

int hw_simd() const {
switch (hw) {
case ngen::HW::Gen9:
case ngen::HW::Gen10:
case ngen::HW::Gen11:
case ngen::HW::XeLP:
case ngen::HW::XeHP:
case ngen::HW::XeHPG: return 8;
case ngen::HW::XeHPC: return 16;
default: ir_error_not_expected();
}
return -1;
}

ngen::HW hw;
int regs; 
int reg_size; 

int reg_bank_stride = -1;

std::vector<uint64_t> bank_masks;
std::vector<uint64_t> bundle_masks;
};

struct reg_mask_t {
reg_mask_t() = default;

reg_mask_t(const hw_context_t *hw_ctx, uint64_t chunk_mask = -1)
: hw_ctx(hw_ctx), nchunks(hw_ctx->regs / chunk_bits) {
for (int i = 0; i < nchunks; i++)
chunks[i] = chunk_mask;
}

bool none() const {
uint64_t mask = 0;
for (int i = 0; i < nchunks; i++)
mask |= chunks[i];
return mask == 0;
}

bool test(int i) const {
int ichunk = i / chunk_bits;
int bit = i % chunk_bits;
return (chunks[ichunk] >> bit) & 0x1;
}

bool is_unset(int off, int len) const;

void set(int i, bool value = true) {
int ichunk = i / chunk_bits;
int bit = i % chunk_bits;
if (value)
chunks[ichunk] |= (1ull << bit);
else
chunks[ichunk] &= ~(1ull << bit);
}

void set(int off, int len, bool value = true) {
for (int i = off; i < off + len; i++)
set(i, value);
}

void reset() {
for (int i = 0; i < nchunks; i++)
chunks[i] = 0;
}

int count() const {
int ret = 0;
for (int i = 0; i < nchunks; i++)
ret += ngen::utils::popcnt(chunks[i]);
return ret;
}

int bsf() const {
for (int i = 0; i < hw_ctx->regs; i++) {
if (test(i)) return i;
}
return -1;
}

int bsr() const {
UNUSED(&reg_mask_t::bsr);
for (int i = hw_ctx->regs - 1; i >= 0; i--) {
if (test(i)) return i;
}
return -1;
}

reg_mask_t range_mask(int off, int len) const {
reg_mask_t ret(hw_ctx);
ret = ret << (hw_ctx->regs - len);
ret = ret >> (hw_ctx->regs - len - off);
return ret;
}

int bank() const;

void subtract(const reg_mask_t &other) { *this &= ~other; }

reg_mask_t &operator&=(const reg_mask_t &other) {
for (int i = 0; i < nchunks; i++)
chunks[i] = chunks[i] & other.chunks[i];
return *this;
}

reg_mask_t &operator|=(const reg_mask_t &other) {
UNUSED(&reg_mask_t::operator|=);
for (int i = 0; i < nchunks; i++)
chunks[i] = chunks[i] | other.chunks[i];
return *this;
}

reg_mask_t operator<<(int shift) const {
int idx = shift / chunk_bits;
int bit = shift % chunk_bits;
reg_mask_t ret(hw_ctx, 0);
for (int i = idx + 1; i < nchunks; i++) {
auto c0 = (chunks[i - idx] << bit);
auto c1 = (bit == 0 ? 0
: (chunks[i - idx - 1] >> (chunk_bits - bit)));
ret.chunks[i] = c0 | c1;
}
ret.chunks[idx] = (chunks[0] << bit);
return ret;
}

reg_mask_t operator>>(int shift) const {
int idx = shift / chunk_bits;
int bit = shift % chunk_bits;
reg_mask_t ret(hw_ctx, 0);
for (int i = 0; i + idx + 1 < nchunks; i++) {
auto c0 = (chunks[i + idx] >> bit);
auto c1 = (bit == 0 ? 0
: (chunks[i + idx + 1] << (chunk_bits - bit)));
ret.chunks[i] = c0 | c1;
}
ret.chunks[nchunks - idx - 1] = (chunks[nchunks - 1] >> bit);
return ret;
}

bool operator==(const reg_mask_t &other) const {
for (int i = 0; i < nchunks; i++)
if (chunks[i] != other.chunks[i]) return false;
return true;
}

reg_mask_t operator~() const {
reg_mask_t ret(hw_ctx);
for (int i = 0; i < nchunks; i++)
ret.chunks[i] = ~chunks[i];
return ret;
}

std::string str() const {
UNUSED(&reg_mask_t::str);
std::ostringstream oss;
for (int i = hw_ctx->regs - 1; i >= 0; i--) {
oss << (test(i) ? "1" : "0");
}
return oss.str();
}

IR_DEFINE_DUMP()

static const int chunk_bits = 64;
static const int max_regs = 256;
static const int max_nchunks = max_regs / chunk_bits;

const hw_context_t *hw_ctx = nullptr;
int nchunks = 0;
uint64_t chunks[max_nchunks] = {0};
};

inline reg_mask_t operator&(const reg_mask_t &a, const reg_mask_t &b) {
auto ret = a;
return ret &= b;
}

inline bool reg_mask_t::is_unset(int off, int len) const {
auto m = range_mask(off, len);
return m == (*this & m);
}

inline int reg_mask_t::bank() const {
if (*this == (*this & reg_mask_t(hw_ctx, hw_ctx->bank_masks[0]))) return 0;
if (*this == (*this & reg_mask_t(hw_ctx, hw_ctx->bank_masks[1]))) return 1;
return -1;
}

struct reg_block_mask_t {
reg_block_mask_t() = default;

reg_block_mask_t(const hw_context_t *hw_ctx, int regs) : regs(regs) {
masks.reserve(regs);
for (int i = 0; i < regs; i++)
masks.emplace_back(hw_ctx);

auto &mask0 = masks[0];

int step = hw_ctx->reg_bank_stride;
for (int i = 0; i < hw_ctx->regs; i += step) {
for (int j = i + 1; j < i + step; j++) {
mask0.set(j, false);
}
}
for (int i = hw_ctx->regs - regs + 1; i < hw_ctx->regs; i++) {
mask0.set(i, false);
}
propagate_masks();
}

void exclude(const reg_mask_t &mask) {
for (auto &m : masks)
m.subtract(mask);
}

bool can_be_assigned() const {
for (auto &m : masks)
if (m.none()) return false;
return true;
}

bool is_assigned() const { return masks[0].count() == 1; }

void propagate_masks() {
for (int j = 1; j < regs; j++) {
masks[0] &= (masks[j] >> j);
}
for (int j = 1; j < regs; j++) {
masks[j] &= (masks[0] << j);
}
}

std::string str() const {
UNUSED(&reg_block_mask_t::str);
std::ostringstream oss;
for (int i = 0; i < regs; i++) {
oss << "#" << i << " mask: " << masks[i].str();
if (i != regs - 1) oss << std::endl;
}
return oss.str();
}

IR_DEFINE_DUMP()

int regs;
std::vector<reg_mask_t> masks;
};

struct reg_t {
reg_t() = default;

reg_t(reg_block_mask_t *block, int off) : block(block), off(off) {}

bool is_empty() const { return !block; }

int bank() const {
if (is_empty()) return -1;
return block->masks[off].bank();
}

void exclude(const reg_mask_t &mask) {
if (is_empty()) return;
block->masks[off].subtract(mask);
}

bool operator==(const reg_t &other) const {
return (other.block == block) && (other.off == off);
}

std::string str() const {
UNUSED(&reg_t::str);
if (is_empty()) return "null";
std::ostringstream oss;
if (block->is_assigned()) {
int reg = block->masks[off].bsf();
oss << "r" << reg;
} else {
oss << "R" << off;
}
return oss.str();
}

IR_DEFINE_DUMP()

reg_block_mask_t *block = nullptr;
int off = -1;
};

struct reg_buf_mask_t {
reg_buf_mask_t(const hw_context_t *hw_ctx, int regs, int block_regs = 0)
: hw_ctx(hw_ctx), regs(regs), block_regs(block_regs) {
if (block_regs == 0) this->block_regs = regs;
ir_assert(regs % this->block_regs == 0);
for (int i = 0; i < nblocks(); i++) {
blocks.emplace_back(hw_ctx, this->block_regs);
}
}

int size() const { return regs * hw_ctx->reg_size; }

int nblocks() const { return regs / block_regs; }

reg_t get_reg(int off_bytes) {
ir_assert(off_bytes < size());
off_bytes /= hw_ctx->reg_size;
int block_idx = off_bytes / block_regs;
int reg_idx = off_bytes % block_regs;
return reg_t(&blocks[block_idx], reg_idx);
}

const hw_context_t *hw_ctx;
int regs; 
int block_regs; 
std::vector<reg_block_mask_t> blocks;
};

struct instruction_t {
instruction_t(const reg_t &src0, const reg_t &src1, const reg_t &src2)
: src0(src0), src1(src1), src2(src2) {}

bool has(const reg_t &reg) const {
return reg == src0 || reg == src1 || reg == src2;
}

reg_t src0;
reg_t src1;
reg_t src2;
};

struct search_context_t {
search_context_t(const hw_context_t *hw_ctx, const reg_mask_t &reg_mask,
std::vector<reg_block_mask_t *> &blocks,
const std::vector<instruction_t> &instructions)
: hw_ctx(hw_ctx)
, reg_mask(reg_mask)
, blocks(blocks)
, instructions(instructions) {
saved_blocks.resize(nblocks() * nblocks());
}

int nblocks() { return int(blocks.size()); }

void set_check_bundles(bool value = true) { check_bundles = value; }

void set_check_diff_banks_src02(bool value = true) {
check_diff_banks_src02 = value;
}

void save_blocks() {
ir_assert(saved_block_idx + nblocks() <= int(saved_blocks.size()));
for (int i = 0; i < nblocks(); i++) {
saved_blocks[saved_block_idx + i] = *blocks[i];
}
saved_block_idx += nblocks();
steps++;
}

void restore_blocks() {
saved_block_idx -= nblocks();
ir_assert(saved_block_idx >= 0);
for (int i = 0; i < nblocks(); i++) {
*blocks[i] = saved_blocks[saved_block_idx + i];
}
}

bool should_stop() const {
int max_steps = 250;
return steps > max_steps;
}

void reset_steps() { steps = 0; }

const hw_context_t *hw_ctx;

int steps = 0;
reg_mask_t reg_mask;
std::vector<reg_block_mask_t *> blocks;
std::vector<instruction_t> instructions;

int saved_block_idx = 0;
std::vector<reg_block_mask_t> saved_blocks;

bool check_bundles = false;

bool check_diff_banks_src02 = false;
};

bool search(search_context_t &ctx, int block_idx = 0) {
if (block_idx >= ctx.nblocks()) return true;

auto *hw_ctx = ctx.hw_ctx;

auto &block = *ctx.blocks[block_idx];
auto &mask0 = block.masks[0];

for (int k = 0; k < ctx.hw_ctx->regs; k++) {
int i = block.regs > 4 && hw_ctx->hw <= ngen::HW::XeLP
? ctx.hw_ctx->regs - k - block.regs
: k;
if (!mask0.test(i)) continue;
if (!ctx.reg_mask.is_unset(i, block.regs)) continue;
if (ctx.should_stop()) return false;

ctx.save_blocks();

ctx.reg_mask.set(i, block.regs, false);
reg_mask_t i_reg_mask(hw_ctx, 0);
for (int j = 0; j < block.regs; j++) {
block.masks[j].reset();
block.masks[j].set(i + j);
i_reg_mask.set(i + j);
}

bool conflicts_ok = true;

for (int j = block_idx + 1; j < ctx.nblocks(); j++) {
ctx.blocks[j]->exclude(i_reg_mask);
if (!ctx.blocks[j]->can_be_assigned()) {
conflicts_ok = false;
break;
}
}

std::vector<reg_mask_t> bundle_masks;
bundle_masks.reserve(block.regs);
for (int j = 0; j < block.regs; j++) {
int bundle = hw_ctx->reg_bundle(i + j);
bundle_masks.emplace_back(hw_ctx, hw_ctx->bundle_masks[bundle]);
}

for (auto &insn : ctx.instructions) {
for (int j = 0; j < block.regs; j++) {
reg_t j_reg(&block, j);
if (!insn.has(j_reg)) continue;

int bank0 = insn.src0.bank();
int bank1 = insn.src1.bank();
int bank2 = insn.src2.bank();

if (ctx.check_diff_banks_src02) {
if (bank0 != -1 && bank0 == bank2) {
conflicts_ok = false;
break;
}
}

if (bank0 != -1 && bank0 == bank1 && bank1 == bank2) {
conflicts_ok = false;
break;
}

if (ctx.check_bundles) {
for (auto *reg : {&insn.src0, &insn.src1, &insn.src2}) {
if (*reg == j_reg) continue;
reg->exclude(bundle_masks[j]);
}
}
break;
}
}

if (conflicts_ok) {
for (auto *b : ctx.blocks) {
b->propagate_masks();
if (!b->can_be_assigned()) {
conflicts_ok = false;
break;
}
}
}

if (conflicts_ok) {
bool ok = search(ctx, block_idx + 1);
if (ok) return true;
}

ctx.reg_mask.set(i, block.regs, true);
ctx.restore_blocks();
}
return false;
}

reg_mask_t create_available_reg_mask(
reg_allocator_t &ra, const hw_context_t *hw_ctx) {
reg_mask_t reg_mask(hw_ctx, 0);
ra.start_speculate();

for (;;) {
auto grf = ra.try_alloc();
if (grf.isInvalid()) break;
reg_mask.set(grf.getBase());
}

for (int i = 0; i < hw_ctx->regs; i++) {
if (reg_mask.test(i)) {
ngen::GRF grf(i);
ra.safeRelease(grf);
}
}

ra.finish_speculate();
return reg_mask;
}

} 

bank_conflict_allocation_t bank_conflict_allocation_t::create(
reg_allocator_t &ra, int regs, const bank_conflict_attr_t &attr) {
hw_context_t hw_ctx(ra.hardware(), regs);

bool is_dpas = false;
bool is_dp4a = false;
expr_t dst_base;
if (!attr.instructions.empty()) {
auto &s = attr.instructions[0];
auto &func = s.as<func_call_t>().func;
if (func.is<dpas_t>()) {
is_dp4a = func.as<dpas_t>().is_dp4a();
is_dpas = !is_dp4a;
dst_base = get_base(dpas_t::arg_dst(s));
} else if (func.is<mad_t>()) {
dst_base = get_base(mad_t::arg_dst(s));
} else {
ir_error_not_expected();
}
}

int dst_block_regs = (is_dpas ? 0 : 2);
int src_block_regs = (is_dpas || is_dp4a ? 0 : 16);

std::vector<expr_t> bufs = attr.bufs;
std::vector<reg_buf_mask_t> buf_masks;
std::vector<int> buf_src_idx(bufs.size(), -1);
for (int i = 0; i < int(bufs.size()); i++) {
int buf_size = attr.buf_sizes[i];
int min_block_size = attr.buf_min_block_sizes[i];
int regs = utils::div_up(buf_size, hw_ctx.reg_size);
int block_regs = (bufs[i].is_equal(dst_base) ? dst_block_regs
: src_block_regs);
if (block_regs != 0 && regs % block_regs != 0) block_regs = 0;
if (block_regs != 0 && block_regs * hw_ctx.reg_size < min_block_size)
block_regs = 0;
buf_masks.emplace_back(&hw_ctx, regs, block_regs);
}

auto create_reg = [&](const expr_t &e, int src_idx, int off_bytes) {
if (is_zero(e)) return reg_t();
auto base = get_base(e);
int off = 0;
if (!is_var(e)) off = to_cpp<int>(e.as<ptr_t>().off);
off += off_bytes;
for (size_t i = 0; i < bufs.size(); i++) {
if (base.is_same(bufs[i])) {
buf_src_idx[i] = src_idx;
return buf_masks[i].get_reg(off);
}
}
ir_error_not_expected();
return reg_t();
};

int hw_simd = hw_ctx.hw_simd();
std::vector<instruction_t> instructions;
for (auto &s : attr.instructions) {
auto &call = s.as<func_call_t>();
expr_t src0, src1, src2;
int simd = 0;
int src0_stride_bytes;
int src1_stride_bytes;
int src2_stride_bytes;
if (call.func.is<dpas_t>()) {
auto &dpas = call.func.as<dpas_t>();
simd = dpas.exec_size;
src0_stride_bytes = dpas.dst_type.size();
src1_stride_bytes = dpas.src1_type.size();
src2_stride_bytes = dpas.is_dp4a() ? 0 : dpas.src2_type.size();
src0 = dpas_t::arg_src0(call);
src1 = dpas_t::arg_src1(call);
src2 = dpas_t::arg_src2(call);
if (!dpas.is_dp4a()) ir_assert(simd == hw_simd);
} else if (call.func.is<mad_t>()) {
auto &mad = call.func.as<mad_t>();
simd = mad.exec_size;
src0_stride_bytes = mad.dst_type.size();
src1_stride_bytes = mad.src1_stride * mad.src1_type.size();
src2_stride_bytes = mad.src2_stride * mad.src2_type.size();
src0 = mad_t::arg_src0(call);
src1 = mad_t::arg_src1(call);
src2 = mad_t::arg_src2(call);
} else {
ir_error_not_expected();
}
for (int off = 0; off < simd; off += hw_simd) {
auto _src0 = create_reg(src0, 0, off * src0_stride_bytes);
auto _src1 = create_reg(src1, 1, off * src1_stride_bytes);
auto _src2 = create_reg(src2, 2, off * src2_stride_bytes);
instructions.emplace_back(_src0, _src1, _src2);
}
}

std::vector<reg_block_mask_t *> blocks;

for (size_t i = 0; i < bufs.size(); i++)
ir_assert(buf_src_idx[i] != -1)
<< "Buffer is not referenced: " << bufs[i];

for (int i : {1, 2, 0}) {
for (size_t j = 0; j < bufs.size(); j++) {
if (buf_src_idx[j] == i) {
for (auto &block : buf_masks[j].blocks)
blocks.push_back(&block);
}
}
}

auto reg_mask = create_available_reg_mask(ra, &hw_ctx);
search_context_t ctx(&hw_ctx, reg_mask, blocks, instructions);

if (is_dpas) ctx.set_check_diff_banks_src02();

bool found = false;

for (bool check_bundles : {true, false}) {
if (is_dpas && check_bundles) continue;
if (hw_ctx.hw <= ngen::HW::XeLP && check_bundles) continue;
ctx.reset_steps();
ctx.set_check_bundles(check_bundles);

ir_assert(ctx.saved_block_idx == 0);
ir_assert(ctx.reg_mask == reg_mask);

#ifdef GEN_CONV_DEBUG
double search_time = get_msec();
#endif
found = search(ctx);
#ifdef GEN_CONV_DEBUG
search_time = get_msec() - search_time;
ir_trace() << "Bank conflict allocation:" << std::endl;
ir_trace() << "    Search time: " << search_time << " ms" << std::endl;
ir_trace() << "    Status: " << (found ? "OK" : "FAIL") << std::endl;
ir_trace() << "    Steps: " << ctx.steps << std::endl;
ir_trace() << "    Bundle check: "
<< ir_utils::to_string(ctx.check_bundles) << std::endl;
#endif
if (found) break;
}

bool was_claimed = false;
if (!found) {
int bank = -1;
auto &buf_sizes = attr.buf_sizes;
std::vector<int> idx(bufs.size());
std::iota(idx.begin(), idx.end(), 0);
auto block_cmp = [&](int idx0, int idx1) {
return buf_sizes[idx0] > buf_sizes[idx1];
};
std::sort(idx.begin(), idx.end(), block_cmp);
for (int i : idx) {
bool is_src02 = utils::one_of(buf_src_idx[i], 0, 2);
int regs = buf_masks[i].regs;
buf_masks[i] = reg_buf_mask_t(&hw_ctx, regs);
ngen::Bundle bundle;
if (is_src02 && bank != -1)
bundle = ngen::Bundle(1 - bank, ngen::Bundle::any);
auto &mask = buf_masks[i].blocks[0].masks[0];
auto range = ra.alloc_range(regs, bundle);
int base = range[0].getBase();
if (is_src02 && bank == -1) bank = hw_ctx.reg_bank(base);
mask.reset();
mask.set(base);
}
was_claimed = true;
}

bank_conflict_allocation_t bca(ra);
for (size_t i = 0; i < bufs.size(); i++) {
int nblocks = buf_masks[i].nblocks();
std::vector<int> block_bases(nblocks);
for (int j = 0; j < nblocks; j++) {
int reg = buf_masks[i].blocks[j].masks[0].bsf();
block_bases[j] = reg;
}
reg_buf_t reg_buf(ra.hardware(), buf_masks[i].block_regs, block_bases);
if (!was_claimed) reg_buf.claim(ra);
bca.set_reg_buf(bufs[i], reg_buf);
}
return bca;
}

} 
} 
} 
} 

#if defined(__GNUC__) && __GNUC__ == 7
#pragma GCC diagnostic pop
#endif
