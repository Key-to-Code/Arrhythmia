#!/usr/bin/env python3
"""
export_cnn_weights.py - Convert model_int8.tflite into ecg_cnn_weights.h
(weights + quantization params + self-test vectors) for the bare-metal engine ecg_cnn.c.

Usage:  python export_cnn_weights.py --tflite output/model_int8.tflite --out output/ecg_cnn_weights.h
        (optional) --samples X_test.npy   float beats in [0,1], shape (N,250)

Model it expects (project report, train_ecg_tinyml.py):
    250 -> Conv16 k3 (ReLU) -> BN -> Pool2 -> Conv32 k3 (ReLU) -> BN -> Pool4 -> Dense16 -> Dense3
The engine itself is generic, a different layer stack only triggers a warning.
Exit code is 1 if the numpy twin of the C engine disagrees with TFLite, so a build script can stop.
"""
import argparse
import sys
import numpy as np

COMPUTE = {'CONV_2D': 'conv', 'FULLY_CONNECTED': 'fc', 'MUL': 'mul', 'ADD': 'add', 'MAX_POOL_2D': 'pool'}
LAYOUT_ONLY = {'RESHAPE', 'EXPAND_DIMS', 'SQUEEZE', 'SOFTMAX', 'SHAPE', 'STRIDED_SLICE', 'PACK'}
OPCODE = {'conv': 'CNN_OP_CONV', 'mul': 'CNN_OP_MUL', 'add': 'CNN_OP_ADD', 'pool': 'CNN_OP_POOL', 'fc': 'CNN_OP_FC'}


# ----------------------------------------------------------------- reference engine (mirrors ecg_cnn.c)
def _req(v, zp):
    v = np.clip(np.asarray(v, np.float32), -1000, 1000)
    v = np.trunc(v + np.where(v >= 0, 0.5, -0.5)).astype(np.int32) + int(zp)
    return np.clip(v, -128, 127).astype(np.int8)


def ref_forward(layers, x):
    """Numpy twin of the C engine. x: int8 (250,) -> int8 logits."""
    a = np.asarray(x, np.int8).reshape(-1)
    f32 = np.float32
    for L in layers:
        t = L['type']
        if t in ('conv', 'fc'):
            n = L['k'] * L['in_ch']
            w = L['w'].reshape(L['out_ch'], n).astype(np.int32)
            a32 = a.astype(np.int32)
            m = (f32(L['in_scale']) * L['ws'].astype(f32)) / f32(L['out_scale'])
            out = np.zeros((L['out_len'], L['out_ch']), np.int8)
            for tt in range(L['out_len']):
                xp = a32[tt * L['in_ch']: tt * L['in_ch'] + n]
                acc = L['b'].astype(np.int64) + (w @ xp) - int(L['in_zp']) * w.sum(1)
                out[tt] = _req(acc.astype(f32) * m, L['out_zp'])
            a = out.reshape(-1)
        elif t in ('mul', 'add'):
            C = L['out_ch']
            x2 = a.reshape(-1, C).astype(np.int32) - int(L['in_zp'])
            cv = f32(L['c_scale']) * (L['w'].astype(np.int32) - int(L['c_zp'])).astype(f32)
            if t == 'mul':
                v = x2.astype(f32) * ((f32(L['in_scale']) * cv) / f32(L['out_scale']))
            else:
                v = x2.astype(f32) * (f32(L['in_scale']) / f32(L['out_scale'])) + cv / f32(L['out_scale'])
            a = _req(v, L['out_zp']).reshape(-1)
        elif t == 'pool':
            C, k = L['out_ch'], L['k']
            x2 = a.reshape(-1, C)[:L['out_len'] * k].reshape(L['out_len'], k, C)
            a = x2.max(axis=1).astype(np.int8).reshape(-1)
    return a


def softmax(q, scale, zp):
    f = (q.astype(np.float32) - zp) * scale
    e = np.exp(f - f.max())
    return e / e.sum()


# ----------------------------------------------------------------- model sanity checks
BEAT_LEN, NUM_CLASSES = 250, 3
# (type, out_len, out_ch) of the compute layers in the report's model; BN shows up as MUL+ADD and is skipped here
REPORT_ARCH = [('conv', 248, 16), ('pool', 124, 16), ('conv', 122, 32),
               ('pool', 30, 32), ('fc', 1, 16), ('fc', 1, 3)]


def count_macs(layers):
    return sum(L['out_len'] * L['out_ch'] * L['k'] * L['in_ch'] for L in layers if L['type'] in ('conv', 'fc'))


def check_model(layers):
    """Hard-fail on the I/O contract the firmware depends on, warn if the body differs from the report."""
    n_in = layers[0]['in_len'] * layers[0]['in_ch']
    assert n_in == BEAT_LEN, f"model input is {n_in} samples, firmware feeds {BEAT_LEN}"
    assert layers[-1]['out_len'] * layers[-1]['out_ch'] == NUM_CLASSES, \
        f"model has {layers[-1]['out_ch']} outputs, firmware expects {NUM_CLASSES} classes (N/V/S)"
    got = [(L['type'], L['out_len'], L['out_ch']) for L in layers if L['type'] not in ('mul', 'add')]
    if got != REPORT_ARCH:
        print("[export] NOTE: layer stack differs from the report model")
        print(f"         report: {REPORT_ARCH}")
        print(f"         found : {got}")
    else:
        print("[export] layer stack matches the report model (Conv16-Conv32-Dense16-Dense3)")


def print_footprint(layers):
    wbytes = sum(L['w'].size for L in layers if L['w'] is not None)
    bbytes = sum(L['b'].size * 4 for L in layers if L['b'] is not None)
    sbytes = sum(L['ws'].size * 4 for L in layers if L['ws'] is not None)
    act = max(L['out_len'] * L['out_ch'] for L in layers)
    print(f"[export] flash ~{wbytes + bbytes + sbytes} B (weights {wbytes}, bias {bbytes}, scales {sbytes}), "
          f"RAM {2 * act} B (two ping-pong buffers), {count_macs(layers)} MACs/beat")


# ----------------------------------------------------------------- TFLite extraction
def _ls(shape):
    """(time_length, channels) from a tensor shape, channel-last."""
    s = [int(v) for v in shape]
    return (1, s[-1]) if len(s) <= 2 else (s[-2], s[-1])


def extract_layers(tflite_path):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=tflite_path)
    it.allocate_tensors()
    det = {d['index']: d for d in it.get_tensor_details()}
    produced = {it.get_input_details()[0]['index']}
    layers = []
    for op in it._get_ops_details():
        name = op['op_name']
        outs = op['outputs']
        if name in LAYOUT_ONLY:
            produced.update(outs)
            continue
        if name not in COMPUTE:
            raise RuntimeError(f"Unsupported op in model: {name}. Extend ecg_cnn.c / this exporter.")
        kind = COMPUTE[name]
        ins = [i for i in op['inputs'] if i >= 0]
        o = det[outs[0]]
        L = dict(type=kind, out_scale=o['quantization'][0], out_zp=o['quantization'][1],
                 c_scale=0.0, c_zp=0, w=None, b=None, ws=None, k=1)
        if kind in ('conv', 'fc'):
            iin, iw, ib = ins[:3]
            w = it.get_tensor(iw).astype(np.int8)
            L['b'] = it.get_tensor(ib).astype(np.int32)
            sc = np.asarray(det[iw]['quantization_parameters']['scales'], np.float32)
            nout = w.shape[0]
            L['ws'] = np.full(nout, sc[0], np.float32) if sc.size == 1 else sc
            L['in_scale'], L['in_zp'] = det[iin]['quantization']
            if kind == 'conv':
                assert w.shape[1] == 1, "only 1-D conv (kernel height 1) supported"
                L['k'] = int(w.shape[2])
                L['in_len'], L['in_ch'] = _ls(det[iin]['shape'])
                L['out_len'], L['out_ch'] = _ls(o['shape'])
                assert L['out_len'] == L['in_len'] - L['k'] + 1, "only VALID padding, stride 1"
            else:
                L['in_len'], L['in_ch'], L['k'] = 1, int(w.shape[1]), 1
                L['out_len'], L['out_ch'] = 1, int(w.shape[0])
            L['w'] = w.reshape(-1)
        elif kind in ('mul', 'add'):
            iact = [i for i in ins if i in produced]
            iconst = [i for i in ins if i not in produced]
            assert len(iact) == 1 and len(iconst) == 1, "MUL/ADD must have one constant operand"
            L['w'] = it.get_tensor(iconst[0]).astype(np.int8).reshape(-1)
            L['c_scale'], L['c_zp'] = det[iconst[0]]['quantization']
            L['in_scale'], L['in_zp'] = det[iact[0]]['quantization']
            L['in_len'], L['in_ch'] = _ls(det[iact[0]]['shape'])
            L['out_len'], L['out_ch'] = _ls(o['shape'])
        else:  # pool
            iin = ins[0]
            L['in_scale'], L['in_zp'] = det[iin]['quantization']
            L['in_len'], L['in_ch'] = _ls(det[iin]['shape'])
            L['out_len'], L['out_ch'] = _ls(o['shape'])
            L['k'] = L['in_len'] // L['out_len']
            assert L['in_len'] // L['k'] == L['out_len']
        produced.update(outs)
        layers.append(L)
    # chain consistency
    for a, b in zip(layers, layers[1:]):
        assert a['out_len'] * a['out_ch'] == b['in_len'] * b['in_ch'], "layer size mismatch"
    tfl_out = it.get_output_details()[0]
    return it, layers, tfl_out


# ----------------------------------------------------------------- header writer
def _f(v):
    t = f"{float(v):.9g}"
    return (t if any(c in t for c in '.en') else t + '.0') + 'f'


def _arr(name, a, ctype, per=12):
    flat = np.asarray(a).reshape(-1)
    body = []
    for i in range(0, len(flat), per):
        body.append("  " + ", ".join(
            (_f(v) if ctype == 'float' else str(int(v))) for v in flat[i:i + per]))
    return f"static const {ctype} {name}[{len(flat)}] = {{\n" + ",\n".join(body) + "\n};\n"


def write_header(path, layers, in_q, tfl_out_q, tests):
    """tests: list of (int8 input[250], int8 tflite_output[3])"""
    max_act = max(L['out_len'] * L['out_ch'] for L in layers)
    last = layers[-1]
    o = ["/* AUTO-GENERATED by export_cnn_weights.py - do not edit by hand. */",
         "#ifndef ECG_CNN_WEIGHTS_H\n#define ECG_CNN_WEIGHTS_H\n",
         f"#define CNN_IN_LEN        {layers[0]['in_len'] * layers[0]['in_ch']}",
         f"#define CNN_IN_SCALE      {_f(in_q[0])}", f"#define CNN_IN_ZP         {int(in_q[1])}",
         f"#define CNN_OUT_SCALE     {_f(last['out_scale'])}   /* logits (before softmax) */",
         f"#define CNN_OUT_ZP        {int(last['out_zp'])}",
         f"#define CNN_TFL_OUT_SCALE {_f(tfl_out_q[0])}   /* TFLite softmax output, self-test only */",
         f"#define CNN_TFL_OUT_ZP    {int(tfl_out_q[1])}",
         f"#define CNN_NUM_CLASSES   {last['out_len'] * last['out_ch']}",
         f"#define CNN_NUM_LAYERS    {len(layers)}", f"#define CNN_MAX_ACT       {max_act}",
         f"#define CNN_MACS          {count_macs(layers)}",
         f"#define CNN_NUM_TESTS     {len(tests)}\n"]
    rows = []
    for i, L in enumerate(layers):
        if L['w'] is not None:
            o.append(_arr(f"L{i}_W", L['w'], 'int8_t'))
        if L['b'] is not None:
            o.append(_arr(f"L{i}_B", L['b'], 'int32_t'))
        if L['ws'] is not None:
            o.append(_arr(f"L{i}_WS", L['ws'], 'float', 6))
        p = lambda key, s: f"L{i}_{s}" if L[key] is not None else "0"
        rows.append(f"  {{ {OPCODE[L['type']]}, {L['in_len']}, {L['in_ch']}, {L['out_len']}, {L['out_ch']}, {L['k']}, "
                    f"{_f(L['in_scale'])}, {_f(L['out_scale'])}, {_f(L['c_scale'])}, "
                    f"{int(L['in_zp'])}, {int(L['out_zp'])}, {int(L['c_zp'])}, "
                    f"{p('w','W')}, {p('b','B')}, {p('ws','WS')} }}")
    o.append("static const cnn_layer_t cnn_layers[CNN_NUM_LAYERS] = {\n" + ",\n".join(rows) + "\n};\n")
    o.append("static const int8_t cnn_test_inputs[CNN_NUM_TESTS][CNN_IN_LEN] = {\n" +
             ",\n".join("  {" + ", ".join(str(int(v)) for v in t[0]) + "}" for t in tests) + "\n};\n")
    o.append("static const int8_t cnn_test_expected[CNN_NUM_TESTS][3] = {\n" +
             ",\n".join("  {" + ", ".join(str(int(v)) for v in t[1]) + "}" for t in tests) + "\n};\n")
    o.append("#endif\n")
    with open(path, 'w') as f:
        f.write("\n".join(o))


# ----------------------------------------------------------------- main
def _synthetic_beats(n=3, size=250):
    t = np.linspace(-1, 1, size)
    kinds = [np.exp(-t**2 / (2 * 0.02**2)) + 0.3 * np.exp(-(t - 0.5)**2 / 0.02),   # narrow R + T
             np.exp(-t**2 / (2 * 0.08**2)) - 0.5 * np.exp(-(t - 0.3)**2 / 0.03),   # wide QRS
             0.2 * np.exp(-(t + 0.3)**2 / 0.01) + np.exp(-t**2 / (2 * 0.03**2))]   # P + QRS
    out = []
    for k in kinds[:n]:
        k = k + 0.01 * np.random.randn(size)
        out.append((k - k.min()) / (k.max() - k.min()))
    return np.array(out, np.float32)


def export_weights(tflite_path, out_path, samples=None, n_tests=6):
    it, layers, tfl_out = extract_layers(tflite_path)
    check_model(layers)
    inp = it.get_input_details()[0]
    in_q = inp['quantization']
    if samples is None:
        np.random.seed(0)                      # same header every run
        samples = _synthetic_beats()
    samples = np.asarray(samples, np.float32)[:max(n_tests, 1)].reshape(-1, 250)
    tests, agree, maxdiff = [], 0, 0.0
    for x in samples:
        q = _req(x / np.float32(in_q[0]), in_q[1])      # same rounding as ecg_cnn_quantize() in C
        it.set_tensor(inp['index'], q.reshape(inp['shape']))
        it.invoke()
        tout = it.get_tensor(tfl_out['index'])[0].astype(np.int8)
        mine = softmax(ref_forward(layers, q), layers[-1]['out_scale'], layers[-1]['out_zp'])
        ref = softmax(tout, *tfl_out['quantization'])
        agree += int(np.argmax(mine) == np.argmax(ref))
        maxdiff = max(maxdiff, float(np.abs(mine - ref).max()))
        tests.append((q, tout))
    print(f"[export] layers: {[L['type'] for L in layers]}")
    print_footprint(layers)
    print(f"[export] python engine vs TFLite: argmax agreement {agree}/{len(tests)}, max prob diff {maxdiff:.4f}")
    bad = maxdiff > 0.05
    if bad:
        print("[export] WARNING: engine deviates from TFLite - check op list above before flashing!")
    write_header(out_path, layers, in_q, tfl_out['quantization'], tests)
    print(f"[export] wrote {out_path}")
    return not bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", default="./output/model_int8.tflite")
    ap.add_argument("--out", default="./output/ecg_cnn_weights.h")
    ap.add_argument("--samples", default=None, help=".npy of normalized beats (N,250) for self-test vectors")
    a = ap.parse_args()
    ok = export_weights(a.tflite, a.out, np.load(a.samples) if a.samples else None)
    sys.exit(0 if ok else 1)
