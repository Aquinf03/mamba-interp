# L17 key-matched value patch

Same keys; donor has a different value for the queried key. Patch donor L17 h at last write onto the clean sequence.

| n | clean acc | donor-seq acc | still original V | **now donor V** |
|---:|---:|---:|---:|---:|
| 128 | 0.984 | 1.000 | 0.984 | 0.000 |

If **now donor V** >> chance and original V drops, L17 h is content-addressable for that key.
