"""Audited, TensorFlow-free reader for the original dc01 measurement records.

Tensor encodings are read according to TensorFlow Example/TensorProto schemas.
Physical panel assignments come from the publisher's spec, not a fabricated
256x4 competition-shaped array. Frequency calibration follows the accompanying
NVlabs reference: fftshift, then positive CPO/STO compensation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import numpy as np

RECORD_BYTES = 525022
RECORD_COUNT = 32133
RAW_MD5 = "7934b83963491b493a37556484d6ca42"


def fields(data):
    pos = 0

    def varint():
        nonlocal pos
        value, shift = 0, 0
        while True:
            if pos >= len(data) or shift > 63:
                raise ValueError("Malformed protobuf varint")
            byte = data[pos]
            pos += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value
            shift += 7

    while pos < len(data):
        tag = varint()
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            yield field, wire, varint()
        elif wire in (1, 5):
            n = 8 if wire == 1 else 4
            value = data[pos:pos+n]
            if len(value) != n:
                raise ValueError("Truncated fixed-width field")
            pos += n
            yield field, wire, value
        elif wire == 2:
            n = varint()
            value = data[pos:pos+n]
            if len(value) != n:
                raise ValueError("Truncated length-delimited field")
            pos += n
            yield field, wire, value
        else:
            raise ValueError(f"Unsupported wire type {wire}")


def only(data, number):
    result = [value for key, _, value in fields(data) if key == number]
    if len(result) != 1:
        raise ValueError(f"Expected exactly one protobuf field {number}")
    return result[0]


def tensor(data):
    values = {key: value for key, _, value in fields(data)}
    dtype = {1: np.dtype("<f4"), 2: np.dtype("<f8")}[values[1]]
    shape = []
    for key, _, value in fields(values[2]):
        if key == 2:
            shape.append(int(only(value, 1)))
    return np.frombuffer(values[4], dtype=dtype).reshape(shape)


def example(data):
    result = {}
    for key, _, entry in fields(only(data, 1)):
        if key != 1:
            continue
        name = bytes(only(entry, 1)).decode("utf-8")
        feature = only(entry, 2)
        kind, _, value = next(fields(feature))
        if kind == 1:
            result[name] = tensor(only(value, 1))
        elif kind == 2:
            packed = only(value, 1)
            result[name] = float(np.frombuffer(packed, dtype="<f4")[0])
        else:
            raise ValueError(f"Unexpected Feature type {kind}")
    return result


def crc32c(data):
    # Used on small header samples; the full file has an independent official MD5.
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ (0x82F63B78 if c & 1 else 0)
        table.append(c)
    value = 0xFFFFFFFF
    for byte in data:
        value = table[(value ^ byte) & 255] ^ (value >> 8)
    return value ^ 0xFFFFFFFF


def masked_crc(data):
    value = crc32c(data)
    return (((value >> 15) | (value << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def read_record(raw, index, verify_crc=False):
    start = index * RECORD_BYTES
    length, header_crc = struct.unpack("<QI", raw[start:start+12])
    if length != RECORD_BYTES-16:
        raise ValueError(f"Record {index} length differs from fixed source layout: {length}")
    payload = memoryview(raw)[start+12:start+12+length]
    if len(payload) != length:
        raise ValueError("Incomplete TFRecord")
    if verify_crc:
        if masked_crc(bytes(raw[start:start+8])) != header_crc:
            raise ValueError("TFRecord length CRC mismatch")
        expected = struct.unpack("<I", raw[start+12+length:start+16+length])[0]
        if masked_crc(payload) != expected:
            raise ValueError("TFRecord payload CRC mismatch")
    return example(payload)


def coordinate_frame(root):
    with (root / "context/data/coordinates.csv").open() as stream:
        rows = {row["Name"]: row for row in csv.DictReader(stream)}
    def xyz(name):
        row = rows[name]
        return np.array([float(row[k]) for k in ("West", "South", "Height")])
    origin = xyz("AU")
    vector = xyz("NWU")-origin
    angle = np.pi/2-np.arctan2(vector[1], vector[0])
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return origin, rotation


def atomic_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--inspect-part", type=Path)
    parser.add_argument("--voxel-m", type=float, default=0.20)
    parser.add_argument("--maximum-points", type=int, default=6000)
    parser.add_argument("--frequency-stride", type=int, default=4)
    args = parser.parse_args()
    started = time.time()
    root = args.data_dir
    if args.inspect_part:
        blob = args.inspect_part.read_bytes()
        x = read_record(blob, 0, verify_crc=True)
        print(json.dumps({k: {"shape": list(v.shape), "dtype": str(v.dtype),
                              "min": float(np.min(v)), "max": float(np.max(v))}
                          if isinstance(v, np.ndarray) else v for k, v in x.items()}, indent=2))
        return
    out = args.output_dir
    if out is None:
        raise ValueError("--output-dir is required for a full audit")
    out.mkdir(parents=True, exist_ok=True)
    verified = json.loads((root / "download_verified.json").read_text())
    path = root / "dichasus-dc01.tfrecords"
    if verified["md5"] != RAW_MD5 or path.stat().st_size != RECORD_BYTES*RECORD_COUNT:
        raise ValueError("Source download has not passed the official identity check")
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    positions = np.empty((RECORD_COUNT, 3), np.float64)
    timestamps = np.empty(RECORD_COUNT, np.float64)
    ages = np.empty(RECORD_COUNT, np.float64)
    snr = np.empty((RECORD_COUNT, 64), np.float32)
    cfo = np.empty_like(snr)
    for i in range(RECORD_COUNT):
        record = read_record(raw, i, verify_crc=i in (0, RECORD_COUNT//2, RECORD_COUNT-1))
        if record["csi"].shape != (64, 1024, 2):
            raise ValueError(f"Unexpected channel shape in record {i}")
        if not np.isfinite(record['csi']).all():
            raise ValueError(f'Non-finite original measured CSI in record {i}')
        positions[i] = record["pos-tachy"]
        timestamps[i] = record["time"]
        ages[i] = record["gt-interp-age-tachy"]
        snr[i] = record["snr"]
        cfo[i] = record["cfo"]
        if i % 4096 == 0:
            print(json.dumps({"stage": "metadata", "record": i}), flush=True)
    np.savez(out / "all_metadata.npz", position=positions, time=timestamps, age=ages, snr=snr, cfo=cfo)
    if not np.isfinite(positions).all():
        raise ValueError("Non-finite positions")
    # Bind all snapshots of a spatial cell before making any label-dependent decision.
    cell = np.floor(positions[:, :2] / args.voxel_m).astype(np.int64)
    _, groups = np.unique(cell, axis=0, return_inverse=True)
    centers = (cell + 0.5)*args.voxel_m
    distance = np.sum((positions[:, :2]-centers)**2, axis=1)
    ordered = np.lexsort((np.arange(RECORD_COUNT), distance, groups))
    first = np.r_[True, groups[ordered][1:] != groups[ordered][:-1]]
    selected = ordered[first]
    available = len(selected)
    if len(selected) > args.maximum_points:
        selected = np.random.default_rng(20261015).choice(selected, args.maximum_points, replace=False)
    selected.sort()
    spec = json.loads((root / "spec.json").read_text())
    offsets = json.loads((root / "context/data/reftx-offsets-dichasus-dc01.json").read_text())
    assignments = np.asarray([a["assignments"] for a in spec["antennas"]])
    if sorted(assignments.ravel().tolist()) != list(range(64)):
        raise ValueError("Invalid physical antenna mapping")
    frequencies = np.arange(0, 1024, args.frequency_stride)
    correction = np.exp(1j*(np.asarray(offsets["cpo"])[:, None] +
                            2*np.pi*np.asarray(offsets["sto"])[:, None]*np.arange(1024)/1024)).astype(np.complex64)
    channel_tmp = out / "channels.partial.npy"
    channel = np.lib.format.open_memmap(channel_tmp, mode="w+", dtype=np.complex64,
                                        shape=(len(selected), 2, 8, 4, len(frequencies)))
    powers = []
    for j, i in enumerate(selected):
        z = read_record(raw, int(i))["csi"]
        h = np.fft.fftshift(z[..., 0] + 1j*z[..., 1], axes=1).astype(np.complex64)*correction
        if not np.isfinite(h).all():
            raise ValueError(f"Non-finite measured CSI in record {i}")
        h = h[assignments][:, :, :, frequencies].transpose(0, 2, 1, 3)
        channel[j] = h
        powers.append(float(np.mean(np.abs(h)**2)))
    channel.flush()
    del channel
    os.replace(channel_tmp, out / "channels.npy")
    origin, rotation = coordinate_frame(root)
    converted = (positions[selected]-origin) @ rotation.T
    panel_positions = (np.array([a["location"] for a in spec["antennas"]])-origin) @ rotation.T
    directions = np.array([a["direction"] for a in spec["antennas"]]) @ rotation.T
    np.savez(out / "selection.npz", source_id=selected, position=converted, original_position=positions[selected],
             time=timestamps[selected], group=groups[selected], panel_position=panel_positions,
             panel_direction=directions, frequency_indices=frequencies, origin=origin, rotation=rotation)
    report = dict(status="complete", source_records=RECORD_COUNT, unique_voxels=available,
                  selected_points=len(selected), voxel_m=args.voxel_m, frequency_stride=args.frequency_stride,
                  channel_shape=[len(selected), 2, 8, 4, len(frequencies)], dtype="complex64",
                  frequency_hz=spec["frequency"], occupied_bandwidth_hz=spec["bandwidth"],
                  physical_panels=2, antennas_per_panel=[4,8], ue_transmit_ports=1,
                  source_md5=RAW_MD5, source_sha256=verified["sha256"],
                  cpo_sto_source="publisher calibration mirrored by accompanying NVlabs repository",
                  panel_assignments_unique=True, all_source_csi_finite=True,
                  zero_power=int(np.sum(np.array(powers)==0)),
                  timestamp_range=[float(timestamps.min()),float(timestamps.max())],
                  position_min=positions.min(axis=0).tolist(),position_max=positions.max(axis=0).tolist(),
                  age_quantiles=np.quantile(ages,[0,.5,.9,.99,1]).tolist(),
                  snr_quantiles=np.quantile(snr,[0,.1,.5,.9,1]).tolist(),
                  power_quantiles=np.quantile(powers,[0,.1,.5,.9,1]).tolist(),
                  elapsed_s=time.time()-started,
                  limitations=["Published RX calibration does not establish TX absolute phase/time coherence",
                               "Two physical RX panels are not two UE transmit ports",
                               "Frequency subsampling preserves band span but reduces unambiguous delay window"])
    preparation=dict(source_sha256=verified['sha256'],
                     code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     spec_sha256=hashlib.sha256((root/'spec.json').read_bytes()).hexdigest(),
                     offsets_sha256=hashlib.sha256((root/'context/data/reftx-offsets-dichasus-dc01.json').read_bytes()).hexdigest(),
                     coordinates_sha256=hashlib.sha256((root/'context/data/coordinates.csv').read_bytes()).hexdigest(),
                     voxel_m=args.voxel_m,maximum_points=args.maximum_points,frequency_stride=args.frequency_stride)
    report['preparation']=preparation
    report['preparation_sha256']=hashlib.sha256(json.dumps(preparation,sort_keys=True).encode()).hexdigest()
    atomic_json(out / "data_audit.json",report)
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
