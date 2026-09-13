# 01_data — gathering spine

Build order: 01 → 02 → 02b → 03 → 04 → 05 → 06 → 06b → 07 → 08 →
(refresh_sp400_membership as needed).

All scripts write to `../01_data/db.h5` (HDF5). Every script documents
its keys and refresh semantics in its header. Read docs/quirks.md BEFORE
fighting any provider issue — the traps are all catalogued.

First build downloads full price history (~1-2 h). Subsequent nightly
refreshes are incremental. Keys in ../.env.
