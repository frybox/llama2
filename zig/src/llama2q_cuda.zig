const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const assert = std.debug.assert;
const print = std.debug.print;


comptime {
  @setFloatMode(.optimized);
}


// ---------------------------------------------------------------------------
// CUDA runtime, as used from the host. Device addresses are CUdeviceptr
// values (unsigned long long == u64 here); a `float*`/`int8_t*` C parameter
// is ABI-identical to u64, so the launchers below are declared with u64
// params and called with u64 device addresses. The default stream is 0
// (what the launchers in src/llama2q_cuda.cu use internally).
//
// Quantized (Q8) port of llama2_cuda: activations and weights are int8 +
// per-group scales; {token,pos} travel to device through a gparams buffer
// read from inside the kernels, so the forward pass is a fixed sequence of
// launches (no per-token re-launch configuration needed).
// ---------------------------------------------------------------------------

const CUresult = c_int;
const CUdeviceptr = u64;
const cudaMemcpyHostToDevice: c_int = 1;
const cudaMemcpyDeviceToHost: c_int = 2;

extern "C" fn cudaMalloc(p: *CUdeviceptr, size: usize) CUresult;
extern "C" fn cudaFree(p: CUdeviceptr) CUresult;
extern "C" fn cudaMemcpy(dst: CUdeviceptr, src: CUdeviceptr, size: usize, kind: c_int) CUresult;
extern "C" fn cudaGetErrorString(e: CUresult) ?[*:0]const u8;
extern "C" fn llama2q_cuda_alloc_host(bytes: usize) ?*anyopaque;
extern "C" fn llama2q_cuda_free_host(p: ?*anyopaque) void;

// CUDA graph handles are opaque pointers in the C++ world; passed as u64 here
// (the build_graph launcher writes them through *CUdeviceptr out-params).

fn checkR(res: CUresult, line: usize) void {
  if (res != 0) {
    const s = cudaGetErrorString(res) orelse "?";
    print("CUDA error: {s} (line {d})\n", .{ s, line });
    std.process.exit(1);
  }
}

// size of f32 in bytes, for device-space pointer arithmetic.
const F32SZ: u64 = 4;


// ---------------------------------------------------------------------------
// Kernel launchers, defined in src/llama2q_cuda.cu (nvcc-compiled object
// linked into this executable). One per kernel; grid/block/shared mirror
// the C forward() call sites.
// ---------------------------------------------------------------------------

extern "C" fn llama2q_cuda_load_emb(x: CUdeviceptr, embeddings: CUdeviceptr, dim: c_int, gp: CUdeviceptr, stream: CUdeviceptr) void;
extern "C" fn llama2q_cuda_rmsnorm_quant(o: CUdeviceptr, x: CUdeviceptr, w: CUdeviceptr, size: c_int, q: CUdeviceptr, s: CUdeviceptr, gs: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2q_cuda_qmatmul_qkv(q: CUdeviceptr, k: CUdeviceptr, v: CUdeviceptr, wq: CUdeviceptr, wk: CUdeviceptr, wv: CUdeviceptr, wsq: CUdeviceptr, wks: CUdeviceptr, wsv: CUdeviceptr, xq: CUdeviceptr, xs: CUdeviceptr, dim: c_int, kvdim: c_int, gp: CUdeviceptr, gs: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2q_cuda_attn_rope_score(attn: CUdeviceptr, q: CUdeviceptr, kcache: CUdeviceptr, gp: CUdeviceptr, nheads: c_int, ncontext: c_int, hsize: c_int, kvdim: c_int, kvmul: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2q_cuda_softmax_rows(attn: CUdeviceptr, gp: CUdeviceptr, ncontext: c_int, nheads: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2q_cuda_attn_value(x1: CUdeviceptr, attn: CUdeviceptr, vcache: CUdeviceptr, gp: CUdeviceptr, nheads: c_int, ncontext: c_int, hsize: c_int, kvdim: c_int, kvmul: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2q_cuda_quantize(x: CUdeviceptr, n: c_int, q: CUdeviceptr, s: CUdeviceptr, gs: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2q_cuda_qmatmul(o: CUdeviceptr, wq: CUdeviceptr, ws: CUdeviceptr, xq: CUdeviceptr, xs: CUdeviceptr, n: c_int, d: c_int, gs: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2q_cuda_axpy_rmsnorm_quant(o: CUdeviceptr, x: CUdeviceptr, y: CUdeviceptr, w: CUdeviceptr, size: c_int, q: CUdeviceptr, s: CUdeviceptr, gs: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2q_cuda_qmatmul_w1w3(h: CUdeviceptr, h1: CUdeviceptr, w1: CUdeviceptr, w3: CUdeviceptr, ws1: CUdeviceptr, ws3: CUdeviceptr, xq: CUdeviceptr, xs: CUdeviceptr, dim: c_int, ffndim: c_int, gs: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2q_cuda_silu_mul_quant(h: CUdeviceptr, h1: CUdeviceptr, ffndim: c_int, q: CUdeviceptr, s: CUdeviceptr, gs: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2q_cuda_axpy(o: CUdeviceptr, x: CUdeviceptr, dim: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2q_cuda_sample_prep(e: CUdeviceptr, idx: CUdeviceptr, s_out: CUdeviceptr, logits: CUdeviceptr, n: c_int, inv_temp: f32, stream: CUdeviceptr) void;
extern "C" fn llama2q_cuda_sample_sort(e: CUdeviceptr, idx: CUdeviceptr, n: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2q_cuda_sample_pick(next: CUdeviceptr, e: CUdeviceptr, idx: CUdeviceptr, n: c_int, topp: f32, coin: f32, s_out: CUdeviceptr, stream: CUdeviceptr) void;

// CUDA-graph build/launch/destroy, defined in the .cu. build_graph captures
// the entire forward (19 launchers) plus a leading H2D memcpy of the pinned
// gparams into one graph; the per-token {token,pos} then travel in through
// gparams_h (pinned host) read at launch time, so forward() is a single
// graph launch and no per-kernel re-launch configuration is needed.
//
// The GraphParams extern struct must mirror llama2q_graph_params in the .cu
// field-for-field (identical C ABI layout: 36 u64 pointer slots then 10 i32).
const GraphParams = extern struct {
  out_exec: *CUdeviceptr,
  out_graph: *CUdeviceptr,
  gparams_h: CUdeviceptr,
  x: CUdeviceptr,
  x1: CUdeviceptr,
  x2: CUdeviceptr,
  h: CUdeviceptr,
  h1: CUdeviceptr,
  q: CUdeviceptr,
  attn: CUdeviceptr,
  logits: CUdeviceptr,
  kcache: CUdeviceptr,
  vcache: CUdeviceptr,
  xq: CUdeviceptr,
  xq_s: CUdeviceptr,
  hq: CUdeviceptr,
  hq_s: CUdeviceptr,
  gparams: CUdeviceptr,
  embeddings: CUdeviceptr,
  wrmsattn: CUdeviceptr,
  wrmsffn: CUdeviceptr,
  wrmsfinal: CUdeviceptr,
  wq: CUdeviceptr,
  wk: CUdeviceptr,
  wv: CUdeviceptr,
  wo: CUdeviceptr,
  w1: CUdeviceptr,
  w2: CUdeviceptr,
  w3: CUdeviceptr,
  wq_s: CUdeviceptr,
  wk_s: CUdeviceptr,
  wv_s: CUdeviceptr,
  wo_s: CUdeviceptr,
  w1_s: CUdeviceptr,
  w2_s: CUdeviceptr,
  w3_s: CUdeviceptr,
  qtok: CUdeviceptr,
  qtok_s: CUdeviceptr,
  dim: c_int,
  kvdim: c_int,
  ffndim: c_int,
  nheads: c_int,
  ncontext: c_int,
  hsize: c_int,
  kvmul: c_int,
  gs: c_int,
  nlayers: c_int,
  nvocab: c_int,
};

extern "C" fn llama2q_cuda_build_graph(p: *const GraphParams) void;
extern "C" fn llama2q_cuda_graph_launch(exec: CUdeviceptr) void;
extern "C" fn llama2q_cuda_graph_destroy(exec: CUdeviceptr, graph: CUdeviceptr) void;


// ---------------------------------------------------------------------------
// Config / host weights / device state — same checkpoint layout as the C.
// ---------------------------------------------------------------------------

const RawConfig = extern struct {
  const Self = @This();
  magic: u32,
  version: i32,
  dim: i32,
  ffndim: i32,
  nlayers: i32,
  nheads: i32,
  nkvheads: i32,
  nvocab: i32,
  ncontext: i32,

  // group_size is read separately (the file packs it right after a u8
  // shared_classifier at byte 37, which does not 4-align inside a struct).
  fn cook(self: Self, group_size: i32) Config {
    return Config{
      .dim = @intCast(self.dim),
      .ffndim = @intCast(self.ffndim),
      .nlayers = @intCast(self.nlayers),
      .nheads = @intCast(self.nheads),
      .nkvheads = @intCast(self.nkvheads),
      .nvocab = @intCast(self.nvocab),
      .ncontext = @intCast(self.ncontext),
      .gs = @intCast(group_size),
    };
  }
};


const Config = struct {
  dim: usize,
  ffndim: usize,
  nlayers: usize,
  nheads: usize,
  nkvheads: usize,
  nvocab: usize,
  ncontext: usize,
  gs: usize,

  pub fn kvdim(self: Config) usize {
    return self.dim * self.nkvheads / self.nheads;
  }
};


// Host-side view over the mmap'd checkpoint (weights start at byte 256).
// Quantized tensors are per-layer interleaved in the file:
// [l0.q][l0.s][l1.q][l1.s]... where each layer holds `size_each` int8s
// followed by `size_each / gs` f32 scales.
const Weights = struct {
  const Self = @This();
  gs: usize,
  wrmsattn: [*]const f32,
  wrmsffn: [*]const f32,
  wrmsfinal: [*]const f32,
  qtok: [*]const i8,
  qtok_s: [*]const f32,
  // per-tensor file base (layer-0 q); layer l sits at
  // base + l * (size_each + size_each / gs * 4) bytes.
  wq: [*]const i8,
  wk: [*]const i8,
  wv: [*]const i8,
  wo: [*]const i8,
  w1: [*]const i8,
  w2: [*]const i8,
  w3: [*]const i8,

  // bytes of one layer's (q + s) block for each tensor
  fn stride(gs: usize, size_each: usize) usize {
    return size_each + size_each / gs * 4;
  }

  fn layerQ(gs: usize, size_each: usize, l: u64) u64 {
    return l * @as(u64, @intCast(stride(gs, size_each)));
  }

  fn init(c: *const Config, data: []u8) Self {
    const dim = c.dim;
    const ffndim = c.ffndim;
    const nlayers = c.nlayers;
    const gs = c.gs;
    const kvdim = dim * c.nkvheads / c.nheads;
    const dim2: u64 = @intCast(dim * dim);
    const dimkv: u64 = @intCast(dim * kvdim);
    const dimff: u64 = @intCast(dim * ffndim);
    const ffn_dim: u64 = @intCast(ffndim * dim);
    const ntok: u64 = @intCast(c.nvocab * dim);

    var w: Self = undefined;
    w.gs = gs;
    var p: [*]u8 = data.ptr;
    w.wrmsattn = @ptrCast(@alignCast(p)); p += nlayers * dim * 4;
    w.wrmsffn = @ptrCast(@alignCast(p)); p += nlayers * dim * 4;
    w.wrmsfinal = @ptrCast(@alignCast(p)); p += dim * 4;
    w.qtok = @ptrCast(@alignCast(p)); p += ntok;
    w.qtok_s = @ptrCast(@alignCast(p)); p += ntok / gs * 4;
    w.wq = @ptrCast(@alignCast(p)); p += nlayers * @as(u64, @intCast(stride(gs, dim2)));
    w.wk = @ptrCast(@alignCast(p)); p += nlayers * @as(u64, @intCast(stride(gs, dimkv)));
    w.wv = @ptrCast(@alignCast(p)); p += nlayers * @as(u64, @intCast(stride(gs, dimkv)));
    w.wo = @ptrCast(@alignCast(p)); p += nlayers * @as(u64, @intCast(stride(gs, dim2)));
    w.w1 = @ptrCast(@alignCast(p)); p += nlayers * @as(u64, @intCast(stride(gs, dimff)));
    w.w2 = @ptrCast(@alignCast(p)); p += nlayers * @as(u64, @intCast(stride(gs, ffn_dim)));
    w.w3 = @ptrCast(@alignCast(p));
    return w;
  }
};


fn initBuf(comptime T: type, n: usize) CUdeviceptr {
  var ptr: CUdeviceptr = 0;
  checkR(cudaMalloc(&ptr, @sizeOf(T) * n), @src().line);
  if (ptr == 0) {
    print("cudaMalloc state failed\n", .{});
    std.process.exit(1);
  }
  return ptr;
}

fn freeBuf(ptr: CUdeviceptr) void {
  _ = cudaFree(ptr);
}

fn copyToDev(dst: CUdeviceptr, src: usize, bytes: usize) void {
  checkR(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), @src().line);
}


// Device weights.
const DevWeights = struct {
  const Self = @This();
  embeddings: CUdeviceptr,
  wrmsattn: CUdeviceptr,
  wrmsffn: CUdeviceptr,
  wrmsfinal: CUdeviceptr,
  wq: CUdeviceptr,
  wk: CUdeviceptr,
  wv: CUdeviceptr,
  wo: CUdeviceptr,
  w1: CUdeviceptr,
  w2: CUdeviceptr,
  w3: CUdeviceptr,
  wq_s: CUdeviceptr,
  wk_s: CUdeviceptr,
  wv_s: CUdeviceptr,
  wo_s: CUdeviceptr,
  w1_s: CUdeviceptr,
  w2_s: CUdeviceptr,
  w3_s: CUdeviceptr,
  qtok: CUdeviceptr,
  qtok_s: CUdeviceptr,

  fn upload(c: *const Config, w: *const Weights) !Self {
    const dim = c.dim;
    const ffndim = c.ffndim;
    const gs = c.gs;
    const nlayers = c.nlayers;
    const kvdim = c.kvdim();
    const dim2: usize = dim * dim;
    const dimkv: usize = dim * kvdim;
    const dimff: usize = dim * ffndim;
    const ffn_dim: usize = ffndim * dim;
    const ntok: usize = c.nvocab * dim;

    var d: Self = undefined;
    d.embeddings = initBuf(f32, ntok);
    d.wrmsattn = initBuf(f32, nlayers * dim);
    d.wrmsffn = initBuf(f32, nlayers * dim);
    d.wrmsfinal = initBuf(f32, dim);
    d.wq = initBuf(i8, nlayers * dim2);
    d.wq_s = initBuf(f32, nlayers * dim2 / gs);
    d.wk = initBuf(i8, nlayers * dimkv);
    d.wk_s = initBuf(f32, nlayers * dimkv / gs);
    d.wv = initBuf(i8, nlayers * dimkv);
    d.wv_s = initBuf(f32, nlayers * dimkv / gs);
    d.wo = initBuf(i8, nlayers * dim2);
    d.wo_s = initBuf(f32, nlayers * dim2 / gs);
    d.w1 = initBuf(i8, nlayers * dimff);
    d.w1_s = initBuf(f32, nlayers * dimff / gs);
    d.w2 = initBuf(i8, nlayers * ffn_dim);
    d.w2_s = initBuf(f32, nlayers * ffn_dim / gs);
    d.w3 = initBuf(i8, nlayers * dimff);
    d.w3_s = initBuf(f32, nlayers * dimff / gs);
    d.qtok = initBuf(i8, ntok);
    d.qtok_s = initBuf(f32, ntok / gs);

    // embeddings: dequantize qtok on host (emb[i] = q[i] * s[i / gs]).
    const emb = try std.heap.page_allocator.alloc(f32, ntok);
    defer std.heap.page_allocator.free(emb);
    for (0..ntok) |i| {
      emb[i] = @as(f32, @floatFromInt(w.qtok[i])) * w.qtok_s[i / gs];
    }
    copyToDev(d.embeddings, @intFromPtr(emb.ptr), ntok * @sizeOf(f32));

    copyToDev(d.wrmsattn, @intFromPtr(w.wrmsattn), nlayers * dim * @sizeOf(f32));
    copyToDev(d.wrmsffn, @intFromPtr(w.wrmsffn), nlayers * dim * @sizeOf(f32));
    copyToDev(d.wrmsfinal, @intFromPtr(w.wrmsfinal), dim * @sizeOf(f32));
    copyToDev(d.qtok, @intFromPtr(w.qtok), ntok * @sizeOf(i8));
    copyToDev(d.qtok_s, @intFromPtr(w.qtok_s), ntok / gs * @sizeOf(f32));

    // per-layer interleaved q/s in the file -> contiguous per-layer slices.
    // NOTE: *_s device buffers hold f32 scales, so their per-layer byte stride
    // is (size/gs)*4 — multiply by 4, unlike the int8 *_q buffers.
    for (0..nlayers) |l| {
      const li: u64 = @intCast(l);
      copyToDev(d.wq + li * dim2, @intFromPtr(w.wq + Weights.layerQ(gs, dim2, l)), dim2 * @sizeOf(i8));
      copyToDev(d.wq_s + li * dim2 / gs * 4, @intFromPtr(w.wq + Weights.layerQ(gs, dim2, l) + dim2), dim2 / gs * @sizeOf(f32));
      copyToDev(d.wk + li * dimkv, @intFromPtr(w.wk + Weights.layerQ(gs, dimkv, l)), dimkv * @sizeOf(i8));
      copyToDev(d.wk_s + li * dimkv / gs * 4, @intFromPtr(w.wk + Weights.layerQ(gs, dimkv, l) + dimkv), dimkv / gs * @sizeOf(f32));
      copyToDev(d.wv + li * dimkv, @intFromPtr(w.wv + Weights.layerQ(gs, dimkv, l)), dimkv * @sizeOf(i8));
      copyToDev(d.wv_s + li * dimkv / gs * 4, @intFromPtr(w.wv + Weights.layerQ(gs, dimkv, l) + dimkv), dimkv / gs * @sizeOf(f32));
      copyToDev(d.wo + li * dim2, @intFromPtr(w.wo + Weights.layerQ(gs, dim2, l)), dim2 * @sizeOf(i8));
      copyToDev(d.wo_s + li * dim2 / gs * 4, @intFromPtr(w.wo + Weights.layerQ(gs, dim2, l) + dim2), dim2 / gs * @sizeOf(f32));
      copyToDev(d.w1 + li * dimff, @intFromPtr(w.w1 + Weights.layerQ(gs, dimff, l)), dimff * @sizeOf(i8));
      copyToDev(d.w1_s + li * dimff / gs * 4, @intFromPtr(w.w1 + Weights.layerQ(gs, dimff, l) + dimff), dimff / gs * @sizeOf(f32));
      copyToDev(d.w2 + li * ffn_dim, @intFromPtr(w.w2 + Weights.layerQ(gs, ffn_dim, l)), ffn_dim * @sizeOf(i8));
      copyToDev(d.w2_s + li * ffn_dim / gs * 4, @intFromPtr(w.w2 + Weights.layerQ(gs, ffn_dim, l) + ffn_dim), ffn_dim / gs * @sizeOf(f32));
      copyToDev(d.w3 + li * dimff, @intFromPtr(w.w3 + Weights.layerQ(gs, dimff, l)), dimff * @sizeOf(i8));
      copyToDev(d.w3_s + li * dimff / gs * 4, @intFromPtr(w.w3 + Weights.layerQ(gs, dimff, l) + dimff), dimff / gs * @sizeOf(f32));
    }
    return d;
  }

  fn deinit(self: *Self) void {
    freeBuf(self.embeddings);
    freeBuf(self.wrmsattn);
    freeBuf(self.wrmsffn);
    freeBuf(self.wrmsfinal);
    freeBuf(self.wq);
    freeBuf(self.wq_s);
    freeBuf(self.wk);
    freeBuf(self.wk_s);
    freeBuf(self.wv);
    freeBuf(self.wv_s);
    freeBuf(self.wo);
    freeBuf(self.wo_s);
    freeBuf(self.w1);
    freeBuf(self.w1_s);
    freeBuf(self.w2);
    freeBuf(self.w2_s);
    freeBuf(self.w3);
    freeBuf(self.w3_s);
    freeBuf(self.qtok);
    freeBuf(self.qtok_s);
    self.* = undefined;
  }
};


// Device state: activation buffers + sampling workspace.
const State = struct {
  const Self = @This();
  x: CUdeviceptr,
  x1: CUdeviceptr,
  x2: CUdeviceptr,
  h: CUdeviceptr,
  h1: CUdeviceptr,
  q: CUdeviceptr,
  attn: CUdeviceptr,
  logits: CUdeviceptr,
  kcache: CUdeviceptr,
  vcache: CUdeviceptr,
  xq: CUdeviceptr,
  xq_s: CUdeviceptr,
  hq: CUdeviceptr,
  hq_s: CUdeviceptr,
  sample_e: CUdeviceptr,
  sample_idx: CUdeviceptr,
  sample_S: CUdeviceptr,
  next_token: CUdeviceptr,
  gparams: CUdeviceptr,

  fn init(c: *const Config) Self {
    const dim = c.dim;
    const ffndim = c.ffndim;
    const gs = c.gs;
    const kvdim = c.kvdim();
    var s: Self = undefined;
    s.x = initBuf(f32, dim);
    s.x1 = initBuf(f32, dim);
    s.x2 = initBuf(f32, dim);
    s.h = initBuf(f32, ffndim);
    s.h1 = initBuf(f32, ffndim);
    s.q = initBuf(f32, dim);
    s.attn = initBuf(f32, c.nheads * c.ncontext);
    s.logits = initBuf(f32, c.nvocab);
    s.kcache = initBuf(f32, c.nlayers * c.ncontext * kvdim);
    s.vcache = initBuf(f32, c.nlayers * c.ncontext * kvdim);
    s.xq = initBuf(i8, dim);
    s.xq_s = initBuf(f32, dim / gs);
    s.hq = initBuf(i8, ffndim);
    s.hq_s = initBuf(f32, ffndim / gs);
    s.sample_e = initBuf(f32, c.nvocab);
    s.sample_idx = initBuf(i32, c.nvocab);
    s.sample_S = initBuf(f32, 1);
    s.next_token = initBuf(i32, 1);
    s.gparams = initBuf(i32, 2);
    return s;
  }

  fn deinit(self: *Self) void {
    freeBuf(self.x);
    freeBuf(self.x1);
    freeBuf(self.x2);
    freeBuf(self.h);
    freeBuf(self.h1);
    freeBuf(self.q);
    freeBuf(self.attn);
    freeBuf(self.logits);
    freeBuf(self.kcache);
    freeBuf(self.vcache);
    freeBuf(self.xq);
    freeBuf(self.xq_s);
    freeBuf(self.hq);
    freeBuf(self.hq_s);
    freeBuf(self.sample_e);
    freeBuf(self.sample_idx);
    freeBuf(self.sample_S);
    freeBuf(self.next_token);
    freeBuf(self.gparams);
    self.* = undefined;
  }
};


const Transformer = struct {
  const Self = @This();
  c: Config,
  s: State,
  dev_w: DevWeights,
  // CUDA graph of the full forward (captured once at init).
  graph_exec: CUdeviceptr,
  graph: CUdeviceptr,
  gparams_h: CUdeviceptr,

  fn init(c: *const Config, w: *const Weights) !Self {
    var self = Self{
      .c = c.*,
      .s = State.init(c),
      .dev_w = try DevWeights.upload(c, w),
      .graph_exec = 0,
      .graph = 0,
      .gparams_h = 0,
    };
    self.buildGraph();
    return self;
  }

  fn buildGraph(self: *Self) void {
    const c = self.c;
    const dim = c.dim;
    const kvdim = c.kvdim();
    const hsize = dim / c.nheads;
    // pinned host {token,pos} that the graph's first node (an H2D memcpy)
    // reads from at launch time.
    const gph: CUdeviceptr = @intFromPtr(llama2q_cuda_alloc_host(2 * @sizeOf(i32)));
    if (gph == 0) {
      print("cudaHostAlloc failed\n", .{});
      std.process.exit(1);
    }
    self.gparams_h = gph;
    var out_exec: CUdeviceptr = 0;
    var out_graph: CUdeviceptr = 0;
    const p = GraphParams{
      .out_exec = &out_exec,
      .out_graph = &out_graph,
      .gparams_h = gph,
      .x = self.s.x,
      .x1 = self.s.x1,
      .x2 = self.s.x2,
      .h = self.s.h,
      .h1 = self.s.h1,
      .q = self.s.q,
      .attn = self.s.attn,
      .logits = self.s.logits,
      .kcache = self.s.kcache,
      .vcache = self.s.vcache,
      .xq = self.s.xq,
      .xq_s = self.s.xq_s,
      .hq = self.s.hq,
      .hq_s = self.s.hq_s,
      .gparams = self.s.gparams,
      .embeddings = self.dev_w.embeddings,
      .wrmsattn = self.dev_w.wrmsattn,
      .wrmsffn = self.dev_w.wrmsffn,
      .wrmsfinal = self.dev_w.wrmsfinal,
      .wq = self.dev_w.wq,
      .wk = self.dev_w.wk,
      .wv = self.dev_w.wv,
      .wo = self.dev_w.wo,
      .w1 = self.dev_w.w1,
      .w2 = self.dev_w.w2,
      .w3 = self.dev_w.w3,
      .wq_s = self.dev_w.wq_s,
      .wk_s = self.dev_w.wk_s,
      .wv_s = self.dev_w.wv_s,
      .wo_s = self.dev_w.wo_s,
      .w1_s = self.dev_w.w1_s,
      .w2_s = self.dev_w.w2_s,
      .w3_s = self.dev_w.w3_s,
      .qtok = self.dev_w.qtok,
      .qtok_s = self.dev_w.qtok_s,
      .dim = @intCast(dim),
      .kvdim = @intCast(kvdim),
      .ffndim = @intCast(c.ffndim),
      .nheads = @intCast(c.nheads),
      .ncontext = @intCast(c.ncontext),
      .hsize = @intCast(hsize),
      .kvmul = @intCast(c.nheads / c.nkvheads),
      .gs = @intCast(c.gs),
      .nlayers = @intCast(c.nlayers),
      .nvocab = @intCast(c.nvocab),
    };
    llama2q_cuda_build_graph(&p);
    self.graph_exec = out_exec;
    self.graph = out_graph;
    if (out_exec == 0 or out_graph == 0) {
      print("graph capture failed\n", .{});
      std.process.exit(1);
    }
  }

  fn deinit(self: *Self) void {
    if (self.graph_exec != 0) {
      llama2q_cuda_graph_destroy(self.graph_exec, self.graph);
    }
    if (self.gparams_h != 0) {
      llama2q_cuda_free_host(@ptrFromInt(self.gparams_h));
    }
    self.s.deinit();
    self.dev_w.deinit();
  }

  // One forward pass: a single captured-graph launch on the default stream.
  // {token,pos} are written to the pinned host gparams_h, which the graph's
  // first node (an H2D memcpy recorded at capture time) reads at launch.
  fn forward(self: Self, token: usize, pos: usize) void {
    const gpi = @as(*[2]i32, @ptrFromInt(self.gparams_h));
    gpi[0] = @intCast(token);
    gpi[1] = @intCast(pos);
    llama2q_cuda_graph_launch(self.graph_exec);
  }

  // top-p sample from device logits; mirrors sample_device() in the C file
  // (prep -> device sort -> pick, then read back the token). Runs on the
  // default stream after the graph, in submission order.
  fn sampleDevice(self: Self, temperature: f32, topp: f32, coin: f32) usize {
    const s = self.s;
    llama2q_cuda_sample_prep(s.sample_e, s.sample_idx, s.sample_S, s.logits, @intCast(self.c.nvocab), 1.0 / temperature, 0);
    llama2q_cuda_sample_sort(s.sample_e, s.sample_idx, @intCast(self.c.nvocab), 0);
    llama2q_cuda_sample_pick(s.next_token, s.sample_e, s.sample_idx, @intCast(self.c.nvocab), topp, coin, s.sample_S, 0);
    var tok: i32 = 0;
    checkR(cudaMemcpy(@intFromPtr(&tok), s.next_token, @sizeOf(i32), cudaMemcpyDeviceToHost), @src().line);
    return @intCast(tok);
  }
};


// ---------------------------------------------------------------------------
// Tokenizer: identical to the CPU implementation.
// ---------------------------------------------------------------------------

const Tokenizer = struct {
  const Self = @This();
  tokens: [][]u8,
  scores: []f32,
  max_token_len: u32,
  byte_pieces: [512]u8,

  fn fromFile(path: []const u8, vocab_size: usize, allocator: Allocator, io: std.Io) !Self {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    var buf: [4096]u8 = undefined;
    var reader = f.reader(io, &buf);
    return Self.init(&reader.interface, allocator, vocab_size);
  }

  fn init(r: *std.Io.Reader, allocator: Allocator, vocab_size: usize) !Self {
    var tokenizer: Self = undefined;
    for (0..256) |i| {
      tokenizer.byte_pieces[i*2] = @intCast(i);
      tokenizer.byte_pieces[i*2+1] = 0;
    }
    tokenizer.tokens = try allocator.alloc([]u8, vocab_size);
    tokenizer.scores = try allocator.alloc(f32, vocab_size);
    tokenizer.max_token_len = try r.takeInt(@TypeOf(tokenizer.max_token_len), .little);
    for (0..vocab_size) |i| {
      tokenizer.scores[i] = @bitCast(try r.takeInt(u32, .little));
      const token_len = try r.takeInt(u32, .little);
      tokenizer.tokens[i] = try allocator.alloc(u8, token_len);
      try r.readSliceAll(tokenizer.tokens[i]);
    }
    return tokenizer;
  }

  fn deinit(self: *const Self, allocator: Allocator) void {
    for (self.tokens) |t| {
      allocator.free(t);
    }
    allocator.free(self.scores);
    allocator.free(self.tokens);
  }

  fn lookup(self: *const Self, str: []const u8) ?u32 {
    for (self.tokens, 0..) |token, i| {
      if (mem.eql(u8, token, str)) {
        return @intCast(i);
      }
    }
    return null;
  }

  fn encode(self: *const Self, input: []const u8, allocator: Allocator) ![]u32 {
    var tokens = try allocator.alloc(u32, input.len+3);
    const buf = try allocator.alloc(u8, self.max_token_len*2+1+2);
    defer allocator.free(buf);
    var fba_alloc = std.heap.FixedBufferAllocator.init(buf);
    const fba = fba_alloc.allocator();
    var utf8buf: [4]u8 = undefined;
    var ii: usize = 0;
    var ti: usize = 0;
    tokens[ti] = 1; //bos
    ti += 1;
    while (ii < input.len) {
      const l = try std.unicode.utf8ByteSequenceLength(input[ii]);
      const cp: u21 = try std.unicode.utf8Decode(input[ii..][0..l]);
      const el = try std.unicode.utf8Encode(cp, &utf8buf);
      tokens[ti] = self.lookup(utf8buf[0..el]) orelse {
        return error.TokenNotFound;
      };
      ii += l;
      ti += 1;
    }
    while (true) {
      var best_score: f32 = -1e10;
      var best_id: u32 = 0;
      var best_idx: ?usize = null;
      for (0..ti-1) |i| {
        const both = try mem.concat(fba, u8, &[_][]u8 {
            self.tokens[tokens[i]],
            self.tokens[tokens[i+1]],
        });
        defer fba.free(both);
        if (self.lookup(both)) |id| {
          if (self.scores[id] > best_score) {
            best_score = self.scores[id];
            best_id = id;
            best_idx = i;
          }
        }
      }
      if (best_idx) |i| {
        tokens[i] = best_id;
        for (i+1..ti-1) |j| {
          tokens[j] = tokens[j+1];
        }
        ti -= 1;
      } else {
        break;
      }
    }
    tokens = try allocator.realloc(tokens, ti);
    return tokens;
  }

  /// C: char *decode(Tokenizer *t, int prev_token, int token)
  fn decode(self: *const Self, prev_token: u32, token: u32) []const u8 {
    var piece: []const u8 = self.tokens[token];
    if (prev_token == 1 and piece.len > 0 and piece[0] == ' ') piece = piece[1..];
    if (Self.parseBytePiece(piece)) |bytev| {
      const off: usize = @as(usize, bytev) * 2;
      piece = self.byte_pieces[off..off + 1];
    }
    return piece;
  }

  fn parseBytePiece(piece: []const u8) ?u8 {
    if (piece.len != 6 or piece[0] != '<' or piece[1] != '0'
     or piece[2] != 'x' or piece[5] != '>') return null;
    return std.fmt.parseInt(u8, piece[3..5], 16) catch null;
  }
};


// ---------------------------------------------------------------------------
// Sampler: xorshift64* RNG, same seed handling as the C main().
// ---------------------------------------------------------------------------

const Sampler = struct {
  const Self = @This();
  state: u64,

  fn init(seed: u64) Self {
    return Self{ .state = seed };
  }

  fn randomU32(self: *Self) u32 {
    self.state ^= self.state >> 12;
    self.state ^= self.state << 25;
    self.state ^= self.state >> 27;
    const prod: u128 = @as(u128, self.state) * @as(u128, 0x2545F4914F6CDD1D);
    return @truncate(prod >> 32);
  }

  fn randomF32(self: *Self) f32 {
    return @as(f32, @floatFromInt(self.randomU32() >> 8)) / 16777216.0;
  }
};


// ---------------------------------------------------------------------------
// generate: mirrors generate() in c/llama2q_cuda.cu.
// ---------------------------------------------------------------------------

fn generate(transformer: *const Transformer, tokenizer: *const Tokenizer, sampler: *Sampler, steps: usize, stdout: *std.Io.Writer, io: std.Io) void {
  const prompt_tokens: []const u32 = &[_]u32{1}; // just BOS, like the C empty prompt
  const num_prompt_tokens: usize = prompt_tokens.len;
  assert(num_prompt_tokens >= 1);

  var token: usize = prompt_tokens[0];
  var next: usize = 0;
  var pos: usize = 0;
  var timer: ?std.Io.Timestamp = null;
  while (pos < steps) {
    transformer.forward(token, pos);
    if (pos < num_prompt_tokens - 1) {
      next = prompt_tokens[pos + 1];
    } else {
      const coin = sampler.randomF32();
      next = transformer.sampleDevice(1.0, 0.9, coin);
    }
    pos += 1;
    if (next == 1) break;
    const piece = tokenizer.decode(@intCast(token), @intCast(next));
    if (piece.len > 0) {
      stdout.print("{s}", .{piece}) catch {};
      stdout.flush() catch {};
    }
    token = next;
    if (timer == null) {
      timer = std.Io.Clock.awake.now(io);
    }
  }
  stdout.print("\n", .{}) catch {};
  if (pos > 1) {
    const elapsed_ns = timer.?.untilNow(io, .awake).nanoseconds;
    const speed: f64 = @as(f64, @floatFromInt(pos - 1)) * std.time.ns_per_s / @as(f64, @floatFromInt(elapsed_ns));
    print("\ntotal {d} tokens, speed {d:.1} token/s\n\n\n", .{ pos - 1, speed });
  }
}


pub fn main(init: std.process.Init) !void {
  const allocator = init.arena.allocator();
  const io = init.io;

  var stdout_buffer: [4096]u8 = undefined;
  var writer = std.Io.File.stdout().writer(io, &stdout_buffer);
  const stdout: *std.Io.Writer = @ptrCast(&writer.interface);
  defer stdout.flush() catch {};

  // RUNQCUDA_SEED, same as the C main(); otherwise the current nanosecond time.
  const now_ns = std.Io.Clock.real.now(io).nanoseconds;
  var rng_seed: u64 = @truncate(@as(u96, @bitCast(now_ns)));
  if (init.environ_map.get("RUNQCUDA_SEED")) |seed_str| {
    rng_seed = std.fmt.parseInt(u64, seed_str, 10) catch rng_seed;
  }

  const checkpoint_path: []const u8 = "stories15M-q8.bin";
  const tokenizer_path: []const u8 = "tokenizer.bin";
  const steps: usize = 256;

  const checkpoint = try std.Io.Dir.cwd().openFile(io, checkpoint_path, .{});
  defer checkpoint.close(io);
  var buffer: [4096]u8 = undefined;
  var reader = checkpoint.reader(io, &buffer);
  var rawConfig = try reader.interface.takeStruct(RawConfig, .little);
  const file_size = (try checkpoint.stat(io)).size;
  const data = try std.posix.mmap(null, file_size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, checkpoint.handle, 0);
  // group_size is packed at byte 37 (right after a u8 shared_classifier),
  // so read it from the mmap by byte offset, little-endian.
  const gsb: [4]u8 = data[37..41].*;
  const group_size: i32 = @bitCast(@as(u32, gsb[0]) | @as(u32, gsb[1]) << 8 | @as(u32, gsb[2]) << 16 | @as(u32, gsb[3]) << 24);
  const config = rawConfig.cook(group_size);
  print("Config: {any}\n", .{config});
  print("\n", .{});

  const weights = Weights.init(&config, data[256..]);
  var transformer = try Transformer.init(&config, &weights);
  defer transformer.deinit();
  const tokenizer = try Tokenizer.fromFile(tokenizer_path, config.nvocab, allocator, io);
  defer tokenizer.deinit(allocator);
  var sampler = Sampler.init(rng_seed);

  generate(&transformer, &tokenizer, &sampler, steps, stdout, io);
}
