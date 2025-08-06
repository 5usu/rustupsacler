import argparse
import onnx
from onnx import shape_inference

def ensure_dynamic_batch_vi(vi):
    t = vi.type
    if not t.HasField("tensor_type"):
        return
    tt = t.tensor_type
    if not tt.HasField("shape"):
        return
    dims = tt.shape.dim

    # If rank==3 (C,H,W), prepend a batch dim to make it (N,C,H,W)
    if len(dims) == 3:
        new_dims = [onnx.TensorShapeProto.Dimension()] + list(dims)
        new_dims[0].dim_param = "N"
        # Replace dims
        del dims[:]
        dims.extend(new_dims)
        return

    # If rank>=1, make dim0 dynamic ("N")
    if len(dims) >= 1:
        if dims[0].HasField("dim_value"):
            dims[0].ClearField("dim_value")
        if not dims[0].HasField("dim_param") or dims[0].dim_param == "":
            dims[0].dim_param = "N"

def main():
    ap = argparse.ArgumentParser(description="Make batch dimension dynamic; add N to rank-3 outputs.")
    ap.add_argument("src", help="source .onnx")
    ap.add_argument("dst", help="destination .onnx")
    args = ap.parse_args()

    m = onnx.load(args.src)

    # Inputs: ensure dynamic batch (N)
    for vi in m.graph.input:
        ensure_dynamic_batch_vi(vi)

    # Outputs: ensure dynamic batch and rank-4 (prepend N if needed)
    for vi in m.graph.output:
        ensure_dynamic_batch_vi(vi)

    # Optional: re-infer shapes
    try:
        m = shape_inference.infer_shapes(m)
    except Exception as e:
        print(f"[warn] shape inference failed: {e}")

    onnx.checker.check_model(m)
    onnx.save(m, args.dst)
    print(f"Saved: {args.dst}")

if __name__ == "__main__":
    main()

