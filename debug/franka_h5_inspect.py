import glob, h5py, numpy as np

p = sorted(glob.glob("/home/ubuntu/Desktop/jhli/_hdf5_episodes/ep*.h5"))[-1]
print("file:", p)
f = h5py.File(p, "r")

def walk(g, prefix=""):
    for k in g.keys():
        item = g[k]
        path = prefix + "/" + k
        if isinstance(item, h5py.Group):
            walk(item, path)
        else:
            extra = ""
            if item.dtype == object or (item.ndim >= 1 and item.shape[-1:] != ()):
                try:
                    a = item[0]
                    if isinstance(a, (bytes, np.bytes_)):
                        extra = f"  (jpeg bytes, first={len(a)}B)"
                except Exception:
                    pass
            print(f"  {path:48s} shape={str(item.shape):16s} dtype={item.dtype}{extra}")

walk(f)
print("attrs:", dict(f.attrs))
