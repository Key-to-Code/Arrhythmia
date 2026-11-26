"""
export_rr.py - Write rr_model.h from trunk_int8.tflite + float32 fusion head weights.

rr_model.h contains:
  - Quantization parameters for the INT8 trunk (in/out scale and zero-point)
  - Float32 fusion head weights W[18][3] and bias b[3]
  - Self-test vectors (quantised wave input + float RR input + expected class)

"""
import sys
import numpy as np


def _f(v: float) -> str:
    """Format a float32 literal for C source, always with decimal point or exponent."""
    t = f"{float(v):.9g}"
    return (t if any(c in t for c in '.en') else t + '.0') + 'f'


def _arr_f32(name: str, a) -> str:
    flat = np.asarray(a, np.float32).reshape(-1)
    rows = []
    for i in range(0, len(flat), 8):
        rows.append("  " + ", ".join(_f(v) for v in flat[i:i+8]))
    return f"static const float {name}[{len(flat)}] = {{\n" + ",\n".join(rows) + "\n};\n"


def export_rr_weights(
    tflite_path: str,
    out_path: str,
    W1_head,         # (18, 16) float32
    b1_head,         # (16,)    float32
    W2_head,         # (16, 3)  float32
    b2_head,         # (3,)     float32
    samples_wave,    # (N, 250, 1) float32
    samples_rr,      # (N, 2)      float32
    ref_labels,      # (N,)        int (Keras fp32 reference predictions)
    in_scale: float,
    in_zp: int,
    out_scale: float,
    out_zp: int,
) -> bool:
    import tensorflow as tf

    it = tf.lite.Interpreter(
        model_path=tflite_path,
        experimental_op_resolver_type=
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    it.allocate_tensors()
    inp_d = it.get_input_details()[0]
    out_d = it.get_output_details()[0]

    W1_head = np.asarray(W1_head, np.float32)
    b1_head = np.asarray(b1_head, np.float32)
    W2_head = np.asarray(W2_head, np.float32)
    b2_head = np.asarray(b2_head, np.float32)

    tests, agree = [], 0
    for xw, xr, ref in zip(samples_wave, samples_rr, ref_labels):
        # Quantize wave input
        q = np.clip(np.floor(xw.reshape(-1) / in_scale + 0.5) + in_zp,
                    -128, 127).astype('int8')
        it.set_tensor(inp_d['index'], q.reshape(inp_d['shape']))
        it.invoke()
        trunk_q = it.get_tensor(out_d['index'])[0].astype(np.int8)
        trunk_f = (trunk_q.astype(np.float32) - out_zp) * out_scale
        rr_sc   = (xr.reshape(2) - 0.5) * 4.0
        fused   = np.concatenate([trunk_f, rr_sc])
        hidden  = np.maximum(0.0, fused @ W1_head + b1_head)
        logits  = hidden @ W2_head + b2_head
        e = np.exp(logits - logits.max()); probs = e / e.sum()
        pred = int(probs.argmax())
        agree += int(pred == int(ref))
        tests.append((q, xr.astype(np.float32), pred))

    n = len(tests)
    print(f"[export_rr] self-test parity: {agree}/{n} match Keras fp32")
    if agree < n:
        print(f"[export_rr] WARNING: {n - agree} test vector(s) diverged from Keras fp32")

    # Extract INT8 trunk layers so rr_model.h is self-contained for ecg_cnn.c
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
    from export_cnn_weights import COMPUTE, LAYOUT_ONLY, OPCODE, _ls, ref_forward
    det = {d['index']: d for d in it.get_tensor_details()}
    produced = {it.get_input_details()[0]['index']}
    trunk_layers = []
    for op in it._get_ops_details():
        name, outs = op['op_name'], op['outputs']
        if name in LAYOUT_ONLY:
            produced.update(outs)
            continue
        if name not in COMPUTE:
            raise RuntimeError(f"Unsupported op in trunk: {name}")
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
            L['ws'] = np.full(w.shape[0], sc[0], np.float32) if sc.size == 1 else sc
            L['in_scale'], L['in_zp'] = det[iin]['quantization']
            if kind == 'conv':
                L['k'] = int(w.shape[2])
                L['in_len'], L['in_ch'] = _ls(det[iin]['shape'])
                L['out_len'], L['out_ch'] = _ls(o['shape'])
            else:
                L['in_len'], L['in_ch'], L['k'] = 1, int(w.shape[1]), 1
                L['out_len'], L['out_ch'] = 1, int(w.shape[0])
            L['w'] = w.reshape(-1)
        elif kind in ('mul', 'add'):
            iact = [i for i in ins if i in produced]
            iconst = [i for i in ins if i not in produced]
            L['w'] = it.get_tensor(iconst[0]).astype(np.int8).reshape(-1)
            L['c_scale'], L['c_zp'] = det[iconst[0]]['quantization']
            L['in_scale'], L['in_zp'] = det[iact[0]]['quantization']
            L['in_len'], L['in_ch'] = _ls(det[iact[0]]['shape'])
            L['out_len'], L['out_ch'] = _ls(o['shape'])
        else:
            iin = ins[0]
            L['in_scale'], L['in_zp'] = det[iin]['quantization']
            L['in_len'], L['in_ch'] = _ls(det[iin]['shape'])
            L['out_len'], L['out_ch'] = _ls(o['shape'])
            L['k'] = L['in_len'] // L['out_len']
        produced.update(outs)
        trunk_layers.append(L)
    trunk_max_act = max(L['out_len'] * L['out_ch'] for L in trunk_layers)

    # Recompute expected class using ref_forward (exact C-engine twin) so
    # ecg_cnn_selftest() validates bit-exact C firmware execution on the MCU.
    c_twin_tests = []
    for q, xr, tfl_pred in tests:
        trunk_q_ref = ref_forward(trunk_layers, q)
        trunk_f_ref = (trunk_q_ref.astype(np.float32) - out_zp) * out_scale
        rr_sc_ref   = (xr.reshape(2) - 0.5) * 4.0
        fused_ref   = np.concatenate([trunk_f_ref, rr_sc_ref])
        hidden_ref  = np.maximum(0.0, fused_ref @ W1_head + b1_head)
        logits_ref  = hidden_ref @ W2_head + b2_head
        c_pred      = int(logits_ref.argmax())
        c_twin_tests.append((q, xr, c_pred))
    tests = c_twin_tests

    lines = [
        "/* AUTO-GENERATED by export_rr.py, do not edit by hand */",
        "#ifndef RR_MODEL_H\n#define RR_MODEL_H\n",
        "#include \"ecg_cnn.h\"\n",
        "/* Quantization parameters (INT8 CNN trunk) */",
        f"#define RR_IN_SCALE         {_f(in_scale)}",
        f"#define RR_IN_ZP            {int(in_zp)}",
        f"#define RR_TRUNK_OUT_SCALE  {_f(out_scale)}",
        f"#define RR_TRUNK_OUT_ZP     {int(out_zp)}\n",
        "/* Model dimensions */",
        "#define RR_WAVE_LEN         250   /* samples per beat window */",
        "#define RR_TRUNK_FEATS      16    /* CNN trunk output features */",
        "#define RR_RR_FEATS         2     /* RR interval features */",
        "#define RR_FUSED_FEATS      18    /* 16 + 2 */",
        "#define RR_HEAD_HIDDEN      16    /* fusion hidden layer width */",
        "#define RR_NUM_CLASSES      3     /* N=0, V=1, S=2 */",
        f"#define RR_TRUNK_NUM_LAYERS {len(trunk_layers)}",
        f"#define RR_TRUNK_MAX_ACT    {trunk_max_act}",
        f"#define RR_NUM_TESTS        {n}\n",
        "/* INT8 CNN trunk layers (from trunk_int8.tflite) */",
    ]

    def _arr_int(name: str, a, ctype: str, per: int = 12) -> str:
        flat = np.asarray(a).reshape(-1)
        rows = ["  " + ", ".join(str(int(v)) for v in flat[i:i+per]) for i in range(0, len(flat), per)]
        return f"static const {ctype} {name}[{len(flat)}] = {{\n" + ",\n".join(rows) + "\n};\n"

    layer_rows = []
    for i, L in enumerate(trunk_layers):
        if L['w'] is not None:
            lines.append(_arr_int(f"RR_L{i}_W", L['w'], 'int8_t'))
        if L['b'] is not None:
            lines.append(_arr_int(f"RR_L{i}_B", L['b'], 'int32_t'))
        if L['ws'] is not None:
            lines.append(_arr_f32(f"RR_L{i}_WS", L['ws']))
        p = lambda key, s: f"RR_L{i}_{s}" if L[key] is not None else "0"
        layer_rows.append(
            f"  {{ {OPCODE[L['type']]}, {L['in_len']}, {L['in_ch']}, {L['out_len']}, {L['out_ch']}, {L['k']}, "
            f"{_f(L['in_scale'])}, {_f(L['out_scale'])}, {_f(L['c_scale'])}, "
            f"{int(L['in_zp'])}, {int(L['out_zp'])}, {int(L['c_zp'])}, "
            f"{p('w','W')}, {p('b','B')}, {p('ws','WS')} }}"
        )
    lines.append("static const cnn_layer_t rr_trunk_layers[RR_TRUNK_NUM_LAYERS] = {\n" + ",\n".join(layer_rows) + "\n};\n")

    lines.extend([
        "/* Fusion head weights (float32, run on Cortex-M4F FPU) */",
        "/* Layer 1: W1[18][16] + b1[16] -> ReLU */",
        _arr_f32("rr_W1_head", W1_head),
        _arr_f32("rr_b1_head", b1_head),
        "/* Layer 2: W2[16][3] + b2[3] -> Softmax */",
        _arr_f32("rr_W2_head", W2_head),
        _arr_f32("rr_b2_head", b2_head),
        "/* Self-test vectors */",
        "/* Quantised waveform inputs (int8, shape [N][250]) */",
        "static const int8_t rr_test_wave[RR_NUM_TESTS][RR_WAVE_LEN] = {\n" +
        ",\n".join("  {" + ", ".join(str(int(v)) for v in t[0]) + "}" for t in tests) +
        "\n};\n",
        "/* Float RR inputs (float32, shape [N][2]) */",
        "static const float rr_test_rr[RR_NUM_TESTS][RR_RR_FEATS] = {\n" +
        ",\n".join("  {" + _f(t[1][0]) + ", " + _f(t[1][1]) + "}" for t in tests) +
        "\n};\n",
        "/* Expected argmax class: 0=N, 1=V, 2=S */",
        "static const int8_t rr_test_expected[RR_NUM_TESTS] = {" +
        ", ".join(str(t[2]) for t in tests) + "};\n",
        "#endif /* RR_MODEL_H */\n",
    ])

    with open(out_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"[export_rr] wrote → {out_path}")
    return agree == n


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tflite',  required=True, help='Path to trunk_int8.tflite')
    ap.add_argument('--weights', required=True,
                    help='.npz file with keys: W, b, Xw, Xr, y, in_scale, in_zp, out_scale, out_zp')
    ap.add_argument('--out',     default='rr_model.h')
    a = ap.parse_args()
    npz = np.load(a.weights)
    ok = export_rr_weights(
        tflite_path=a.tflite, out_path=a.out,
        W_head=npz['W'], b_head=npz['b'],
        samples_wave=npz['Xw'], samples_rr=npz['Xr'], ref_labels=npz['y'],
        in_scale=float(npz['in_scale']),  in_zp=int(npz['in_zp']),
        out_scale=float(npz['out_scale']), out_zp=int(npz['out_zp']),
    )
    sys.exit(0 if ok else 1)
