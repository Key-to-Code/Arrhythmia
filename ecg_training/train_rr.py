"""
train_rr.py - Train a 3-class hybrid CNN+RR ECG classifier on MIT-BIH.

Outputs written to <output_dir>/:
  ecg_hybrid.keras        Full Keras model (float32, keep for retraining)
  trunk_int8.tflite       INT8 CNN trunk  (deployed to STM32)
  rr_model.h              C header: trunk weights + float32 head + self-test vectors
  test_metrics.json       fp32 / firmware_c accuracy on test set
  manifest.json           Patient split and class-weight provenance

"""
import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from shutil import copyfileobj

import numpy as np
from scipy.signal import resample_poly, sosfilt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Patient-separated splits (201 and 202 are from the same patient, both in test)
SPLITS = {
    "train":      "101 102 106 107 108 109 112 114 115 116 118 119 122 124 203 205 207 208 209 215 220",
    "validation": "104 100 105 111 200 223 230",
    "test":       "103 113 117 121 123 201 202 210 212 213 214 217 219 221 222 228 231 232 233 234",
}
GROUPS = ["NLRej", "VE", "AaJS"]   # 3-class: N, V, S
LABELS = list("NVS")

# RR config (at 256 Hz after resampling from 360 Hz)
RR_NORM    = 512    # normalisation constant in samples (~2 s); abs_byte = gap/512 * 255
RR_HISTORY = 8      # number of recent valid gaps for the moving-average relative feature
RR_MIN     = 1      # any gap <= RR_MIN samples is invalid  → emit neutral [128, 128]
RR_MAX     = 768    # any gap  > RR_MAX samples is invalid  → emit neutral [128, 128]


# Firmware-matched bandpass and 50 Hz notch (matches the MCU filter)
def firmware_sos():
    rows = []
    for kind, hz, q in [('hp', 0.5, 0.707), ('lp', 40.0, 0.707), ('notch', 50.0, 10.0)]:
        angle = 2 * np.pi * hz / 256
        c, alpha = np.cos(angle), np.sin(angle) / (2 * q)
        b = {
            'lp':    [(1 - c) / 2, 1 - c, (1 - c) / 2],
            'hp':    [(1 + c) / 2, -(1 + c), (1 + c) / 2],
            'notch': [1, -2 * c, 1],
        }[kind]
        rows.append(np.array([*b, 1 + alpha, -2 * c, 1 - alpha]) / (1 + alpha))
    return np.asarray(rows)


# RR feature computation
def compute_rr_bytes(sample_indices):
    """
    Given sorted QRS peak positions (256 Hz), returns float32 (N, 2):
      col 0: rr_abs_byte / 255    normalised absolute RR interval
      col 1: rr_rel_byte / 255    relative RR interval vs. recent mean

    First beat and any beat after an invalid gap → neutral (128/255, 128/255).
    History is reset after an invalid gap to avoid carrying stale rhythm context.
    """
    out = np.empty((len(sample_indices), 2), np.float32)
    history = []
    for i, pos in enumerate(sample_indices):
        if i == 0:
            out[i] = 128.0 / 255.0
        else:
            gap = int(sample_indices[i] - sample_indices[i - 1])
            if RR_MIN < gap <= RR_MAX:
                abs_b = float(np.clip(gap / RR_NORM * 255, 0, 255))
                mean_rr = float(np.mean(history)) if history else float(gap)
                rel_b = float(np.clip(gap / mean_rr * 128, 0, 255))
                out[i, 0] = abs_b / 255.0
                out[i, 1] = rel_b / 255.0
                history.append(gap)
                if len(history) > RR_HISTORY:
                    history.pop(0)
            else:
                out[i] = 128.0 / 255.0
                history = []              # Reset history when rhythm context breaks
    return out


# Data loader
def load_split(root: Path, ids: str, download: bool):
    import wfdb
    mapping = {sym: cls for cls, group in enumerate(GROUPS) for sym in group}
    X_wave, X_rr, y_out, records = [], [], [], {}
    sos = firmware_sos()

    for rid in ids.split():
        path = root / rid
        for ext in ('hea', 'dat', 'atr'):
            if not path.with_suffix('.' + ext).is_file():
                if not download:
                    raise FileNotFoundError(
                        f"Record {rid} missing .{ext}. Re-run with --download or "
                        f"point --data_dir to your offline copy."
                    )
                url = f"https://physionet.org/files/mitdb/1.0.0/{rid}.{ext}"
                with NamedTemporaryFile(dir=root, delete=False) as tmp:
                    temp_path = Path(tmp.name)
                    try:
                        with urlopen(url, timeout=60) as r:
                            copyfileobj(r, tmp)
                    except Exception:
                        tmp.close()
                        temp_path.unlink(missing_ok=True)
                        raise
                temp_path.replace(path.with_suffix('.' + ext))

        record = wfdb.rdrecord(str(path))
        ann    = wfdb.rdann(str(path), 'atr')
        if record.fs != 360:
            raise ValueError(f"Record {rid}: unexpected fs={record.fs}, expected 360 Hz")
        lead = record.sig_name.index('MLII') if 'MLII' in record.sig_name else 0
        # Resample 360 → 256 Hz, then apply the firmware-matched filter
        signal = sosfilt(sos, resample_poly(record.p_signal[:, lead], 32, 45))

        beat_positions, beat_waves, beat_labels = [], [], []
        counts, skipped = Counter(), Counter()
        for pos, symbol in zip(ann.sample, ann.symbol):
            if symbol not in mapping:
                skipped[symbol] += 1
                continue
            center = round(pos * 32 / 45)   # annotation position in 256-Hz samples
            if center < 125 or center + 125 > len(signal):
                skipped['boundary'] += 1
                continue
            beat = signal[center - 125: center + 125]  # 250-sample window
            span = float(np.ptp(beat))
            if not np.isfinite(beat).all() or span < 1e-6:
                skipped['invalid_window'] += 1
                continue
            beat_positions.append(center)
            beat_waves.append(((beat - beat.min()) / span).astype('float32'))
            beat_labels.append(mapping[symbol])
            counts[LABELS[mapping[symbol]]] += 1

        rr_feats = compute_rr_bytes(beat_positions)
        X_wave.extend(beat_waves)
        X_rr.extend(rr_feats)
        y_out.extend(beat_labels)
        records[rid] = dict(lead=record.sig_name[lead], counts=counts, skipped=dict(skipped))

    return (
        np.stack(X_wave)[..., None],   # (N, 250, 1)  waveform input
        np.stack(X_rr),                # (N, 2)        RR feature input
        np.asarray(y_out),             # (N,)          integer labels
        records,
    )


# Main entry point
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir',     type=Path,  default=Path('data/mitdb'),
                   help='Directory containing MIT-BIH .hea/.dat/.atr files')
    p.add_argument('--download',     action='store_true',
                   help='Download missing records from PhysioNet automatically')
    p.add_argument('--output_dir',   type=Path,  default=Path('runs/run_v1'),
                   help='Output directory for this training run')
    p.add_argument('--epochs',       type=int,   default=60)
    p.add_argument('--weight_power', type=float, default=0.5,
                   help='Class-weight exponent: 0=none, 0.5=sqrt (default), 1=full inverse-freq')
    p.add_argument('--wave_only',    action='store_true',
                   help='Ablation: fix RR inputs to 128/255 (neutral) to run waveform-only baseline')
    args = p.parse_args()

    import tensorflow as tf
    from tensorflow.keras import layers as L, Model, Input
    tf.keras.utils.set_random_seed(42)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)   # fail if folder already exists → no accidental overwrite

    # Load train, validation, and test splits
    print("=" * 60)
    print("Loading MIT-BIH splits …")
    splits = {k: load_split(args.data_dir, ids, args.download) for k, ids in SPLITS.items()}
    Xw, Xr, y, _  = splits['train']
    Vw, Vr, v, _  = splits['validation']
    Tw, Tr, t, _  = splits['test']
    print(f"  train      : {len(y):,} beats — {Counter(LABELS[i] for i in y)}")
    print(f"  validation : {len(v):,} beats — {Counter(LABELS[i] for i in v)}")
    print(f"  test       : {len(t):,} beats — {Counter(LABELS[i] for i in t)}")

    if args.wave_only:
        Xr = np.full_like(Xr, 128.0 / 255.0)
        Vr = np.full_like(Vr, 128.0 / 255.0)
        Tr = np.full_like(Tr, 128.0 / 255.0)
        print("\n[wave_only] RR inputs fixed to 128/255 (neutral — ablation mode)")

    # Compute inverse-frequency class weights
    counts  = np.bincount(y, minlength=3)
    weights = (len(y) / (3 * counts)) ** args.weight_power
    weights /= np.average(weights, weights=counts)   # mean-normalise so loss magnitude is stable
    print(f"\nClass weights (power={args.weight_power}): N={weights[0]:.3f} V={weights[1]:.3f} S={weights[2]:.3f}")

    # Save split and config manifest
    manifest = dict(
        dataset='PhysioNet mitdb 1.0.0', seed=42, labels=LABELS,
        records={k: s[3] for k, s in splits.items()},
        train_weights=weights.tolist(),
        alignment='reference annotations', fs=256,
        mode='wave_only' if args.wave_only else 'hybrid',
        rr_norm=RR_NORM, rr_history=RR_HISTORY, rr_min=RR_MIN, rr_max=RR_MAX,
    )
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    # Build hybrid CNN + RR model
    wave_in = Input(shape=(250, 1), name='wave')
    rr_in   = Input(shape=(2,),     name='rr')

    # Compact 1D-CNN trunk (Conv16 -> Conv16 -> Dense16)
    x = L.Conv1D(16, 3, activation='relu', name='conv1')(wave_in)
    x = L.BatchNormalization(name='bn1')(x)
    x = L.MaxPooling1D(2, name='pool1')(x)
    x = L.Conv1D(16, 3, activation='relu', name='conv2')(x)
    x = L.BatchNormalization(name='bn2')(x)
    x = L.MaxPooling1D(8, name='pool2')(x)
    x = L.Dropout(0.2, name='drop1')(x)
    x = L.Flatten(name='flat')(x)
    x = L.Dense(16, activation='relu', name='dense16')(x)
    x = L.Dropout(0.2, name='drop2')(x)

    # Fusion: 16 waveform features + 2 scaled RR features = 18-dim vector
    rr_scaled = L.Rescaling(4.0, offset=-2.0, name='rr_scale')(rr_in)
    fused = L.Concatenate(name='fuse')([x, rr_scaled])
    # Output: 2-layer float32 fusion head (runs on STM32F4 FPU, NOT quantized)
    h = L.Dense(16, activation='relu', name='head_hidden')(fused)
    out_probs = L.Dense(3, activation='softmax', name='out')(h)

    model = Model(inputs=[wave_in, rr_in], outputs=out_probs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.summary()

    # Training callbacks
    class ValidationF1(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            pred = self.model.predict([Vw, Vr], batch_size=256, verbose=0).argmax(1)
            f1 = f1_score(v, pred, average='macro', zero_division=0)
            logs['val_macro_f1'] = f1
            print(f"    val_macro_f1 = {f1:.4f}")

    model.fit(
        [Xw, Xr], y,
        validation_data=([Vw, Vr], v),
        batch_size=128,
        epochs=args.epochs,
        class_weight=dict(enumerate(weights)),
        callbacks=[
            ValidationF1(),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_macro_f1', mode='max',
                patience=12, start_from_epoch=5, restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_macro_f1', mode='max',
                factor=0.5, patience=5, min_lr=1e-6,
            ),
        ],
    )
    model.save(out / 'ecg_hybrid.keras')
    print(f"\nModel saved → {out / 'ecg_hybrid.keras'}")

    # Evaluate FP32 model on test set
    print("\nEvaluating fp32 on test set …")
    fp32_preds = model.predict([Tw, Tr], verbose=0).argmax(1)
    results = {
        'fp32': dict(
            report=classification_report(t, fp32_preds, target_names=LABELS,
                                         output_dict=True, zero_division=0),
            confusion=confusion_matrix(t, fp32_preds).tolist(),
        )
    }

    # Quantize and export INT8 CNN trunk
    print("\nQuantizing and exporting INT8 trunk …")
    trunk_model = tf.keras.Model(inputs=wave_in, outputs=x, name='trunk')

    rng = np.random.default_rng(42)
    cal_idx = np.concatenate([
        rng.choice(np.flatnonzero(y == i), min(500, int(counts[i])), replace=False)
        for i in range(3)
    ])

    converter = tf.lite.TFLiteConverter.from_keras_model(trunk_model)
    converter.optimizations            = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset   = lambda: ([Xw[j:j+1]] for j in cal_idx)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type     = tf.int8
    converter.inference_output_type    = tf.int8
    tflite_blob = converter.convert()
    (out / 'trunk_int8.tflite').write_bytes(tflite_blob)
    print(f"TFLite trunk saved → {out / 'trunk_int8.tflite'}")

    # Evaluate firmware_c path (emulates STM32 inference)
    print("\nEvaluating firmware_c path (emulates STM32 inference) …")
    it = tf.lite.Interpreter(model_content=tflite_blob)
    it.allocate_tensors()
    inp_d  = it.get_input_details()[0]
    out_d  = it.get_output_details()[0]
    in_scale,   in_zp   = inp_d['quantization']
    out_scale, out_zp   = out_d['quantization']

    # Float32 2-layer fusion head weights (what goes into rr_model.h)
    h_layer = model.get_layer('head_hidden')
    W1_head = h_layer.get_weights()[0].astype(np.float32)    # (18, 16)
    b1_head = h_layer.get_weights()[1].astype(np.float32)    # (16,)
    head_layer = model.get_layer('out')
    W2_head = head_layer.get_weights()[0].astype(np.float32) # (16, 3)
    b2_head = head_layer.get_weights()[1].astype(np.float32) # (3,)

    def firmware_c_predict(Xw_b, Xr_b):
        """Python emulation of the STM32 two-stage inference:
        1. Quantize wave input → INT8 trunk → dequantize to 16 floats
        2. Concat 2 scaled RR floats → 18→16 ReLU → 16→3 softmax on float32"""
        preds = []
        for xw, xr in zip(Xw_b, Xr_b):
            # Stage 1: INT8 trunk
            q = np.clip(np.floor(xw.reshape(-1) / in_scale + 0.5) + in_zp,
                        -128, 127).astype('int8')
            it.set_tensor(inp_d['index'], q.reshape(inp_d['shape']))
            it.invoke()
            trunk_q = it.get_tensor(out_d['index'])[0]
            trunk_f = (trunk_q.astype(np.float32) - out_zp) * out_scale   # (16,)
            # Stage 2: float32 fusion head
            rr_sc  = (xr.reshape(2) - 0.5) * 4.0
            fused  = np.concatenate([trunk_f, rr_sc])                      # (18,)
            hidden = np.maximum(0.0, fused @ W1_head + b1_head)            # (16,)
            logits = hidden @ W2_head + b2_head                            # (3,)
            e = np.exp(logits - logits.max()); probs = e / e.sum()
            preds.append(int(probs.argmax()))
        return np.array(preds)

    fw_preds = firmware_c_predict(Tw, Tr)
    results['firmware_c'] = dict(
        report=classification_report(t, fw_preds, target_names=LABELS,
                                     output_dict=True, zero_division=0),
        confusion=confusion_matrix(t, fw_preds).tolist(),
    )
    fw_val_preds = firmware_c_predict(Vw, Vr)
    results['firmware_c_val'] = dict(
        report=classification_report(v, fw_val_preds, target_names=LABELS,
                                     output_dict=True, zero_division=0),
    )

    (out / 'test_metrics.json').write_text(json.dumps(results, indent=2))

    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for mode in ('fp32', 'firmware_c'):
        r = results[mode]['report']
        print(f"\n  [{mode}]")
        for cls in ('N', 'V', 'S', 'macro avg'):
            d = r[cls]
            print(f"    {cls:10s}  prec={d['precision']:.3f}  rec={d['recall']:.3f}  f1={d['f1-score']:.3f}  sup={int(d.get('support',0))}")

    # Export C header
    print(f"\nExporting rr_model.h …")
    from export_rr import export_rr_weights
    fp32_val_preds = model.predict([Vw, Vr], batch_size=256, verbose=0).argmax(1)
    sample_idx = np.concatenate([
        np.flatnonzero((v == c) & (fp32_val_preds == c))[:4]
        for c in range(3)
    ])
    if len(sample_idx) < 12:
        sample_idx = np.concatenate([np.flatnonzero(v == c)[:4] for c in range(3)])
    ok = export_rr_weights(
        tflite_path   = str(out / 'trunk_int8.tflite'),
        out_path      = str(out / 'rr_model.h'),
        W1_head=W1_head, b1_head=b1_head,
        W2_head=W2_head, b2_head=b2_head,
        samples_wave  = Vw[sample_idx],
        samples_rr    = Vr[sample_idx],
        ref_labels    = fp32_val_preds[sample_idx],
        in_scale=in_scale,   in_zp=int(in_zp),
        out_scale=out_scale, out_zp=int(out_zp),
    )
    if not ok:
        print("[WARNING] Some self-test vectors failed — check accuracy before flashing!")
    print(f"\nAll outputs in: {out.resolve()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
