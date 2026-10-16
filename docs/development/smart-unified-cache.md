# Smart Unified KV Cache: pool-wide prefix reuse by cell reference counting

## Goal

In `kv_unified` mode the server currently reuses the KV cache **only within a single slot's own
sequence**. A new request picks a slot by LCP similarity (or LRU), and `n_past` is computed as the
common prefix between the new prompt and **that same slot's** `prompt.tokens`. If the best-matching
data lives in a different (possibly busy) slot, the request either blocks or re-processes the
shared prefix from scratch.

This feature makes prefix reuse **pool-wide and content-addressable**, decoupled from slot identity:

1. A new request picks **any free slot** (never blocks waiting for a "best match" slot).
2. The new sequence **claims** matching KV cells token-by-token via reference counting, reusing
   whatever already exists anywhere in the unified pool until no more match is found.
3. Only the non-matching tail is prefilled / generated.
4. The pool never stores duplicate entries for the same `(pos, token)`.

Primary use case: agentic conversations with branched execution. A master conversation splits into
sub-agent conversations. Each branch re-uses the "root" conversation's cells without re-processing
the common tokens, and without being tied to whichever slot the root happened to live in.

## Design decisions (confirmed with the user)

- **Best common prefix selection stays exactly as it is.** The existing `get_available_slot` LCP
  heuristic is unchanged. We do not search every live sequence for the best content match - we use
  the current selection to identify the source sequence, then transplant that source's cells.
- **A new request takes any free slot** and gets its own fresh seq_id. It never blocks just because
  the best-match sequence's slot is busy.
- **The new seq_id claims the source seq_id's cells by reference counting** (`seq_add`). The claim
  is one-way (new borrows from best match).
- **The new seq_id becomes fully independent.** After the claim it holds its own refcount on the
  shared prefix cells, so they survive even if the source seq_id is later evicted. A followup
  request can claim from this new seq_id, and so on - forming a chain of independent owners.
- **Eviction frees a cell only when the last seq_id in the chain releases it** (already guaranteed
  by the cell bitset: `seq_rm` frees only when `seq[i].none()`).

## Why this is feasible (existing primitives)

The low-level machinery is already in place. The gap is entirely at the slot-selection / server
layer.

### 1. Unified mode = one shared pool of cells

In `kv_unified` mode the KV cache is constructed with `n_stream = 1`:

```cpp
// src/llama-kv-cache.cpp
n_stream(unified ? 1 : n_seq_max)
```

so every sequence id maps to the same stream (`seq_to_stream` all `0`), and there is **one**
`llama_kv_cells` object / one K/V tensor pool shared by all sequences. `n_seq_max` is widened to
`LLAMA_MAX_SEQ` (256) in unified mode (`src/llama-context.cpp`).

### 2. Cells already support multi-sequence reference counting

`llama_kv_cells` stores a `seq_set_t` (bitset of `LLAMA_MAX_SEQ`) per cell:

```cpp
// src/llama-kv-cells.h
std::vector<seq_set_t> seq;  // which sequences occupy the i-th cell
```

- `seq_add(i, seq_id)` - add a sequence to a cell (refcount++).
- `seq_rm(i, seq_id)`  - remove a sequence; **the cell is freed only when the last sequence leaves**
  (`if (seq[i].none()) { pos[i] = -1; ... }`). This is exactly the reference-counting primitive the
  feature needs.
- `seq_count(i)`, `seq_has(i, seq_id)` expose the sharing state.

### 3. `seq_cp` already implements same-stream cell sharing

When both sequences are in the same stream (always true in unified mode), `seq_cp` does **not** copy
any K/V data. It just reference-counts:

```cpp
// src/llama-kv-cache.cpp (seq_cp, same-stream branch)
if (cells.seq_has(i, seq_id_src)) {
    cells.seq_add(i, seq_id_dst);
}
```

This is precisely the "claim existing cells for a new sequence" operation. The parent/child
(`copy_prompt_to`) path already uses it.

### 4. The attention mask already honours shared cells

`set_input_kq_mask` includes a cell in a sequence's attention iff `cells.seq_has(j, seq_id)`:

```cpp
// src/llama-kv-cache.cpp
if (!cells.seq_has(j, seq_id)) { goto skip; }
```

So once a cell is claimed by two sequences, both correctly attend to it. The graph indexes cells
by their global cell index and filters by sequence membership - no change needed here.

### 5. Cells already carry the token identity

`apply_ubatch` stores the token in `llama_kv_cell_ext::tok` for text batches:

```cpp
// src/llama-kv-cache.cpp
if (ubatch.is_pos_2d() || ubatch.token || hparams.ple_n_heads > 0) {
    ...
    ext.tok = ubatch.token[i];
    cells.ext_set(idx, ext);
}
```

So `(pos, token)` content matching is possible from the cell metadata alone. (Note:
`has_cell_ext()` currently returns false unless M-RoPE/PLE, so `ext.tok` is not round-tripped through
state save/load - that is fine for in-memory matching.)

## What is missing (the actual work)

### Part A - low-level KV cache

**A1. Content index for prefix lookup.** Add an index in `llama_kv_cache` (or a helper) mapping
`(pos, token)` to cell indices, so a sequence can find the longest contiguous run of cells matching
its token list without scanning the whole pool. A `std::unordered_map<token, std::vector<uint32_t>>`
keyed by token (then filtered by pos) is the simplest first step; a `(pos, token)` key is the
precise one.

**A2. `claim_prefix()` method.** New method on `llama_kv_cache`:

```cpp
// given a new seq_id and the source seq_id whose cells to claim,
// claim the longest common prefix between the new request and the source sequence,
// reference-count the matched cells for seq_id (one-way, independent),
// and return how many tokens were claimed (= the new n_past).
uint32_t claim_prefix(llama_seq_id seq_id,      // the new, independent sequence
                      llama_seq_id seq_src,      // the best-match source sequence
                      const llama_token * tokens,
                      const llama_pos *   pos,
                      uint32_t n_tokens);
```

Internally: walk the new token list, look up each `(pos, token)` against the source sequence's
cells, require a contiguous run whose stored token/pos match, then `seq_add` the new `seq_id` to
each matched cell. Stop at the first non-match. Update the cell's `seq_pos` bookkeeping
(automatic via `seq_add`).

Note: `seq_src` is the *best-match* source identified by the existing LCP heuristic. The claim is
restricted to that one source, not a search across all live sequences.

**A3. Claim-aware `find_slot` / `apply_ubatch`.** Today `apply_ubatch` **overwrites** a
single-sequence cell:

```cpp
// src/llama-kv-cache.cpp (apply_ubatch)
if (!cells.is_empty(idx)) {
    assert(cells.seq_count(idx) == 1);
    cells.rm(idx);           // destroys the other sequence's cell
}
```

For pool-wide sharing, when a cell already holds the matching `(pos, token)` for another sequence,
it must be **shared** (`seq_add`) rather than overwritten (`rm`). Add a "claim" flag/path so that
the prefill of a new sequence reuses matching cells instead of duplicating them. This is the piece
that prevents duplicate entries.

**A4. Eviction correctness.** When a slot/sequence is released, `seq_rm` on its id must not free
cells still referenced by another sequence. The bitset already guarantees this; the server must just
call `seq_rm` (not a blanket clear) on release. Verify the existing release path does this.

### Part B - server integration

**B1. Pick a free slot, never block on "best match".** In `get_available_slot`, the
`slot_prompt_similarity` path only considers non-processing slots, but if no LCP match is found it
falls through to LRU. Ensure a free slot is always selected when one exists (no deferral solely for
cache-reuse reasons).

**B2. Pool-wide claim instead of slot-local `n_past`.** In the prompt-processing loop, replace:

```cpp
n_past = slot.prompt.tokens.get_common_prefix(input_tokens);
```

with a call to `claim_prefix(slot.id, input_tokens, ...)` that searches the whole pool. The claimed
length becomes `n_past`. The slot's own `prompt.tokens` is populated with the claimed prefix so
subsequent reuse (and generation continuation) stays correct.

**B3. Bookkeeping.** The slot's `prompt.tokens` must reflect the claimed prefix so that the
`n_past` / `seq_rm(slot.id, p0, -1)` / `pos_next` logic continues to work. The claimed cells belong
to the new seq_id; the slot's seq_id is added to them via `seq_add`.

### Part C - correctness guards

- **Position mismatch after context shift.** If the root has been shifted (`seq_add` with shift),
  its positions no longer align with a fresh branch at `pos0`. `claim_prefix` must verify `pos`
  matches exactly, and gracefully fall back to no claim when it does not.
- **Multimodal / M-RoPE.** `ext.tok` is not always meaningful for embd batches and 2D positions
  complicate matching. Disable pool-wide claim for `mtmd`/M-RoPE initially (keep the existing
  per-slot behavior).
- **n-gram / PLE architectures.** `get_prev_tokens` rejects batches where a token belongs to more
  than one sequence ("ambiguous history"). Pool-wide claim shares cells across sequences, so these
  architectures must either reject claim or keep claim within a single logical parent.
- **Atomicity with batch prep.** The claim must happen before `prepare()`/`find_slot` so the batch
  only contains the non-claimed tail.

### Part D - DSV4 support (full-prefix copy)

`llama_kv_cache_dsv4` is deliberately per-sequence and does NOT use a unified pool, even when
`--kv-unified` is passed (src/llama-kv-cache-dsv4.cpp: `unified_raw = false`,
`unified_compressed = false`; the `unified` argument is `GGML_UNUSED`). It stacks:

- `kv_raw` (`llama_kv_cache_iswa`) - raw/SWA token cache, per-sequence streams.
- `kv_csa` / `kv_hca` / `kv_lid` (`llama_kv_cache`) - compressed K blocks written by the graph, per-sequence.
- `csa_state` / `hca_state` / `lid_state` (`llama_dsv4_comp_state`) - compressor state, per-stream, NOT reference-counted.

So the reference-counted `seq_claim` model does not apply: source and destination are different
streams, the compressed rows carry no `ext.tok`, and the compressor state can only be copied.
`seq_cp` already handles DSV4 cross-sequence reuse as a full buffer copy (used by parent/child
slots). DSV4 therefore reuses a prefix by **copying the whole source sequence**, which is only
valid when the source is a complete prefix of the request.

`llama_kv_cache_dsv4::seq_claim` implements this: it takes the source length from
`kv_raw->seq_pos_max`, rejects claims longer than the request, verifies the source is a full prefix
via the new read-only `llama_kv_cache::seq_match_prefix`, then calls `seq_cp` and returns the source
length. Partial-prefix sharing is not supported - the compressed state cannot be rolled back
(`seq_pos_min` already reports only the current boundary).

Limitations:
- Partial prefixes (source diverges before its end) are not claimed.
- Cost is a full stream-buffer copy per shared request, not a reference-count bump.
- This fights the upstream `FIXME: currently the cache only supports non-unified mode`. A change
  this invasive to DSV4 should be discussed upstream before submission.

Gotcha found while testing on DSV4 (a multimodal model):
- `slot.prompt.tokens.has_mtmd` is set to `true` for **every** prompt when the server has a
  multimodal context (`server-context.cpp`: `slot.prompt.tokens.has_mtmd = mctx != nullptr`), even
  for pure text. Filtering claim sources on that flag silently disabled the whole feature on vision
  models. The fix is `server_tokens::has_media()`, which checks whether the token list actually
  contains a media chunk (`map_idx_to_media` non-empty).

Verified on `deepseek-v4-flash` (DSV4, multimodal): a request forced to slot 1 with a prompt that
extends slot 0's cached text claims the source prefix (`cache_n 20, prompt_n 5`) and logs
`claimed 20 tokens from slot 1 (source slot 0, shared prefix)`.

## Suggested implementation order (incremental)

1. **[DONE] A2**: add `seq_claim()` to `llama_kv_cache` (plus the `llama_memory_i` default no-op and
   the `llama_memory_seq_claim` public wrapper), with unit tests `claim` and `claim_partial` in
   `tests/llama-context/cases-kv.cpp`. The content index (A1) is deferred - the current
   implementation walks the source's `seq_pos` set with `lower_bound`, which is O(log n) per token
   and sufficient for the first cut.
2. **A3**: make `apply_ubatch`/`find_slot` claim (share) matching cells rather than overwrite, and
   add a unit test proving two sequences can share the same physical cells with no duplicate data.
3. **[DONE] B2 + B1 + B3**: server-side integration. `claim_shared_prefix()` (server-context.cpp)
   finds the source slot with the longest common prefix across the whole pool (including busy slots)
   and claims it for the destination slot via `slot.mem.seq_claim`, only when the destination has no
   own cached prefix. `common_memory::seq_claim()` delegates to the target context only. The
   `--cache-sharing` flag implies `--no-cache-idle-slots` so idle slots stay live as claim sources.
   Verified: a branch request to a fresh slot claims the root's prefix and only processes the
   divergent tail; output matches a non-sharing run exactly.
4. **Part C**: guards for shift, mtmd, n-gram/PLE.
5. **[DONE] Part D**: DSV4 support via full-prefix `seq_cp` copy (see Part D above). Added the
   read-only `llama_kv_cache::seq_match_prefix` and `llama_kv_cache_dsv4::seq_claim`. No server
   change needed - `claim_shared_prefix()` calls `seq_claim` generically.
6. **B4 (optional future)**: content-addressed index as a real secondary structure so claim is O(1)
   per token rather than O(pool).

### Part B limitations / open questions

- **Claim only fires for a destination slot with no own cached prefix.** Claiming over a slot's own
  cached cells would leave it owning both its cells and the overlapping claimed cells (duplicate
  content). The common case (a fresh slot claiming a root's prefix) is covered.
- **Source slot must be idle or share its prefix stably.** Claiming from a busy slot reads its
  already-decoded prefix cells, which is safe, but if that source later shifts/truncates, the claimed
  cells move too. Context-shift safety is a Part C concern.
- **`--cache-idle-slots` is disabled by `--cache-sharing`** because idle-slot clearing would free the
  claim-source cells. If both are wanted, idle slots should instead stay live (not be cleared).

  **Future improvement: make `--cache-sharing` work together with `--cache-idle-slots`.** The current
  coupling is a blunt override - enabling `--cache-sharing` forces `--no-cache-idle-slots`, losing the
  prompt-cache persistence (across server restarts and memory pressure) that idle-slot clearing
  provides. A better design would let both coexist:

  - Treat an idle slot's cells as a **live claim source** instead of clearing them on idle. The slot's
    KV stays in the pool (reference-counted), so a later request can claim its prefix directly. This
    is exactly the pool-wide claim path and avoids the serialize/clear/restore round-trip of the
    prompt cache.
  - Keep the prompt cache for what it is good at: **persisting state across restarts** and **evicting
    under memory pressure**, not as the primary reuse path when `--cache-sharing` is on.
  - Eviction policy (see the sequence-level eviction section) should then decide when an idle slot's
    cells are actually dropped - only when the pool is full and the sequence is low-value (a leaf, not
    a claim source). This replaces the current "clear every idle slot" behavior.
  - The destination-slot restriction (claim only into a slot with no own prefix) also needs relaxing
    so a reused slot can claim a better prefix from the pool instead of keeping its own shorter one.

## Activation flag

The feature is off by default. It is enabled only when **both** are true:

- `--kv-unified` / `--no-kv-unified` (env `LLAMA_ARG_KV_UNIFIED`) - the unified KV pool.
- `--cache-sharing` / `--no-cache-sharing` (env `LLAMA_ARG_CACHE_SHARING`) - pool-wide prefix reuse.

The server throws at init if `--cache-sharing` is set without `--kv-unified` (server-context.cpp,
slot setup). The `--cache-sharing` argument is registered in `common/arg.cpp`, and the field
`common_params::cache_sharing` is declared in `common/common.h`. When `--cache-sharing` is on, the
server also disables `--cache-idle-slots` (server-context.cpp), because clearing idle slots into the
prompt cache would free the claim-source cells that a later request wants to reuse.

## Sequences vs slots: capacity and identity model

There are two distinct limits, currently conflated:

- **Hard ceiling `LLAMA_MAX_SEQ = 256`** (`src/llama-cparams.h`). The per-cell `seq` bitset is
  `std::bitset<LLAMA_MAX_SEQ>`; `seq_idx[]`/`seq_pos[]`/`seq_cpl[]` are `[LLAMA_MAX_SEQ]`;
  `seq_to_stream` is `LLAMA_MAX_SEQ`; `cparams.n_seq_max` throws if > 256. This is the true
  architectural maximum number of distinct seq_ids, independent of slots.
- **Runtime `n_seq_max = n_parallel`** (`common/common.cpp:1714`). The server uses `slot.id`
  (0..n_parallel-1) directly as the seq_id and the batch validates `seq_id < n_seq_max`. So today
  `# sequences == # slots == n_parallel` (1:1 by design).

**In `kv_unified` mode the context already widens the effective seq range to `LLAMA_MAX_SEQ`,**
independent of slots (`src/llama-context.cpp:1849`: `n_seq_max = kv_unified ? LLAMA_MAX_SEQ : n_seq_max`).
The unified cache also has `n_stream = 1` with `seq_to_stream` sized to `LLAMA_MAX_SEQ`, so it
already tolerates up to 256 seq_ids regardless of `n_parallel`.

### Decouple slot (execution unit) from sequence (cache identity)

This is the cleanest model for the feature and directly serves the agentic use case:

- **Slot** = unit of execution/scheduling (a batch row, a sampler, a generation stream). Still
  `n_parallel` of them.
- **Sequence (seq_id)** = unit of cache identity. Up to 256 of them, and many can share cells.
  A branch gets its own seq_id (a persistent logical identity) even though it runs on a pool of
  slots.
- When a slot executes a task, it uses that task's seq_id. The mask (`cells.seq_has(j, seq_id)`)
  resolves which cells the slot attends to, so a slot running seq 47 sees exactly seq 47's cells
  whether or not seq 47 shares cells with seq 3.
- Because a seq_id is a persistent identity, the cache can reference-count cells against it and
  evict cells only when **no** seq_id (root or branch) references them.

**Required changes for "more sequences than slots":**

1. The server maintains a separate seq_id pool (up to 256) decoupled from slot ids. A slot picks up
   whatever seq_id its current task needs instead of `slot.id`.
2. The batch-ext `n_seq_max` must be widened to `LLAMA_MAX_SEQ` in unified mode (the context-level
   check already is; `llama_batch_ext` currently gets `n_parallel` from `llama_n_seq_max`).
3. The mask already handles it - no change needed there.

## Eviction policy (sequence-level, not slot-level)

Today eviction is **slot-centric and destructive** because `slot.id` is the seq_id: reusing a free
slot (`get_free_slots` = empty-first then LRU) destroys that slot's sequence via `apply_ubatch`
(`cells.rm`) or `prompt_clear()` (`seq_rm(id, -1, -1)`). Pool-full retries call
`try_clear_idle_slots()` which also `seq_rm`s idle slots one by one.

In the decoupled model the eviction decision moves from "which slot do I reuse?" to "which
**sequence** do I drop?" - and that is where a value heuristic belongs.

### Why cell-level eviction is already safe

The cell refcount (bitset) makes eviction safe by construction: if branch B claimed seq A's prefix,
dropping A does **not** free the shared cells (B still holds a refcount). Eviction can never corrupt
an active branch. Dropping a sequence only loses its *unique tail* cells and its availability as a
future claim source.

### Proposed eviction policy

1. **Decouple slot selection from eviction.** Slot selection just picks a free (non-processing) slot;
   it destroys nothing. Sequences are parked independently of slots.
2. **Evict only when the pool is full.** Pick a seq_id to drop by value, never by slot reuse.
3. **Never evict** a seq_id with active/queued tasks, or whose cells are referenced by a live branch
   (shared-prefix refcount > 1).
4. **Prefer leaves over roots.** An isolated short branch is low value; a root (master conversation)
   is high value because future branches claim from it.
5. **Prefer LRU** among candidates.
6. **Track ancestry** (A->B when B claims from A) so the policy can identify roots vs branches.

## Open questions to resolve before coding

- **Where does the seq_id pool live and how is a new seq_id allocated?** The server needs a pool of
  up to `LLAMA_MAX_SEQ` seq_ids decoupled from slot ids, with free/reuse tracking.
- **How does a new seq_id get assigned to a slot for execution?** The slot picks up whatever seq_id
  its current task needs; the seq_id is no longer `slot.id`.
- **Does the server still need the parent/child (`copy_prompt_to`/`seq_cp`) mechanism, or does the
  general claim replace it?** Likely the claim subsumes it, but we should verify the child
  scheduling semantics (SLOT_STATE_WAIT_OTHER) still work.
- **Do we need a hard cap on the number of sequences sharing a cell (bitset is 256 wide, fine) or a
  separate refcount for diagnostic/logging?**

## Key files to touch

- `src/llama-kv-cells.h` - already has the refcount primitives; likely no change, or small helper
  additions (e.g. a token index).
- `src/llama-kv-cache.h` / `src/llama-kv-cache.cpp` - `claim_prefix()`, claim-aware
  `find_slot`/`apply_ubatch`, content index.
- `tools/server/server-context.cpp` - slot selection, `n_past` via pool-wide claim, bookkeeping.
- `tests/llama-context/cases-kv.cpp` - tests for sharing / claim / no-duplicate-data.

## Findings and open issues from live testing (DeepSeek-V4-Flash-Vision-Exp)

Test target: `deepseek-v4-flash-0731` (`unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF:UD-Q8_K_XL`,
arch `deepseek4`) served through llama-swap on `:8080`, upstream `:10004`, launched with
`--parallel 2 --kv-unified --cache-ram 8192 --cache-sharing`.

### F1. The prompt cache shadows the claim (main open issue)

`--cache-ram` is on by default and usually wins the race. In `get_available_slot` the server calls
`ret->prompt_save()`, then `ret->prompt_load()` and restores a matching prompt from the prompt cache
**before** `claim_shared_prefix()` runs (which happens later, in `update_slots`). When the restore
succeeds, the destination slot already has its own prefix, so the `n_past_own == 0` gate skips the
claim entirely.

Observed: repeated identical runs produce `cache_n 40/42` on the response and `load: - found better
prompt` in the log, but no new `claimed ... tokens` line. The claim only fired right after model
load, when the prompt cache was empty.

Consequence: today the feature is effectively a **fallback** for prompt-cache misses, not the primary
reuse path. For the agentic branched-execution case the prompt cache already provides cross-slot
reuse (via serialize/restore), so the claim adds little unless it is made primary.

Options:
- **A (recommended).** Prefer a live pool claim over the prompt-cache restore in
  `get_available_slot`. The claim is a direct cell reference-count / buffer copy; the restore is a
  serialize + deserialize round-trip. Hook the selection path so that when a live source with a
  longer LCP exists, the restore is skipped.
- **B.** Allow claiming into a slot that already has its own prefix. Requires a dry-run match
  before `seq_rm`, so the destination's existing prefix is only discarded when a strictly longer
  live prefix is available.
- **C.** Leave as a fallback.

### F2. DSV4 claim verified correct, but a clean baseline is hard to isolate

Right after model load (empty prompt cache) the claim fired cleanly and the output matched a
non-sharing run exactly:

```
seq_claim: seq_id=1 seq_src=0 n_tokens=25 src_len=20 n_base=20 n_swa=20
claimed 20 tokens from slot 1 (source slot 0, shared prefix)
```

A high-entropy check (primes prompt, `n_predict 48`, temp 0, fixed seed) also produced byte-identical
output between the claim/restore run and a reference run. But because the prompt cache restores the
same prefix, the `fresh` reference was never truly fresh, and concurrent agent traffic kept
overwriting the source slot between requests. A fully isolated claim-vs-fresh comparison still needs
a run with `--cache-ram 0` (or the agents paused).

### F3. Claim only fires for an empty destination

`claim_shared_prefix()` returns 0 if the destination slot's `prompt.tokens` shares any prefix with
the request. This avoids the destination owning both its own cells and the overlapping claimed cells
(duplicate content). It also means a reused slot can never improve its prefix from the pool - see
option B in F1.

### F4. `seq_rm` before `seq_claim` is safe only while the F3 gate holds

`claim_shared_prefix()` clears the destination with `slot.mem.seq_rm(slot.id, -1, -1)` **before**
calling `seq_claim`. If `seq_claim` then returns 0 (possible on DSV4, see F5), the destination's
cells have already been dropped. Today this is benign because the F3 gate guarantees the destination
held no matching prefix, so the `seq_rm` frees nothing useful. It becomes a real hazard if the gate
is relaxed (option B): a dry-run match must run before the `seq_rm`.

### F5. DSV4 rejects partial prefixes

`llama_kv_cache_dsv4::seq_claim()` takes the source length from `kv_raw->seq_pos_max` and requires
`src_len <= n_tokens` plus a full match via `llama_kv_cache::seq_match_prefix`. A source whose
sequence diverges before its end cannot be claimed, because the compressor state cannot be rolled
back. So the LCP reported by `get_common_prefix()` can be positive while the claim returns 0. The
server falls through to normal prefill in that case.

### F6. DSV4 keeps its caches per-sequence even under `--kv-unified`

`llama_kv_cache_dsv4` passes `unified_raw = false` / `unified_compressed = false` and ignores the
`unified` argument. So `n_ctx` is effectively **per slot**, and the memory is per sequence.

### F7. KV cache memory sizing (measured on `deepseek-v4-flash-0731`)

Four K-only caches, f16, `head_count_kv = 1`, `head_dim = 512` (43 blocks):

| cache | layers | compression | per-token |
|---|---|---|---|
| CSA compressed | 21 | 1 cell / 4 tok | 5.25 KiB |
| LiD indexer | 21 | 1 cell / 4 tok | 1.31 KiB |
| HCA compressed | 20 | 1 cell / 128 tok | 0.16 KiB |
| SWA raw | 43 | windowed, fixed | ~97 MiB/seq |

Startup allocation at `-c 1m --parallel 2` (from the server log):

```
SWA raw 193.50 MiB, CSA 10752.00 MiB, HCA 320.00 MiB, LiD 2688.00 MiB
=> total 13953.5 MiB (~13.6 GiB)
```

So per sequence: ~6.72 KiB/token + ~97 MiB fixed.

| context | per sequence | `--parallel 2` |
|---|---|---|
| 128k | ~957 MiB | ~1.9 GiB |
| 1M | ~6.8 GiB | ~13.6 GiB |

Prompt-cache (`--cache-ram`) serialized state, measured directly: ~11.8 MiB fixed + ~6.8 KiB/token
asymptotically (base state, before context checkpoints). A saved 128k-token prompt is ~0.9 GiB, so
`--cache-ram 8192` holds roughly 3-5 full 128k prompts including checkpoints, or many more typical
agent-size (10k-80k) prompts.

### F8. Test harness notes

- llama-swap listens on `:8080`; the upstream path is
  `http://127.0.0.1:8080/upstream/deepseek-v4-flash-0731/`. The active log is
  `~/.config/llama-swap/stdout.log`. That log is **shared across all llama-swap models**, so
  prompt_save/cache-state sizes must be correlated by timestamp, not read as one model.
- `/slots/0?action=erase` returns 501 without `--slot-save-path`; there is no easy way to force a
  fresh slot. Use `--cache-ram 0` to isolate the claim path instead.
- Concurrent agents hitting the model overwrite slot state between test requests; pause them for
  clean A/B runs.
