# ONNX Light optimization coverage

This inventory compares all **129** names registered by ONNX Light at
[`43c2ac6aa6a393dfb7b90527638d0c1f22139c1e`][registry] with ONNX Script.
It is a comparison of optimization ideas, not a port of its implementation.
The inspected [LICENSE][license] and [NOTICE][notice] specify Apache-2.0;
the Python implementations here are independently written using ONNX Script's
existing IR, matcher, and rewrite conventions.

**Added** means that a sound subset is available through the default
`onnxscript.rewriter.rewrite` and `onnxscript.optimizer.optimize` pipelines.
**Existing** identifies an existing equivalent, including explicitly opt-in
fusion surfaces where noted. **Not adopted** identifies an actual semantic,
backend, or scope-of-optimizer reason, not a promise of future coverage.
These statuses do not imply that every syntactic variant accepted upstream is safe.

Unless otherwise qualified, target module names below are in
`onnxscript/rewriter/rules/common/`. Constant folding refers to
`onnxscript/optimizer/_constant_folding.py`; CSE refers to the IR
`CommonSubexpressionEliminationPass` already run by `optimizer.optimize`.
New rules preserve shared producers and graph outputs, or conservatively reject
a match when deleting its intermediates would affect another use. They do not
raise the model's imported opset. Unknown dimensions are not treated as equal.
Initializer defaults on graph inputs remain overridable, not compile-time constants.
Existing ONNX/ORT rotary and attention fusions also recognize the resulting Split,
repeated-trigonometry, singleton-axis, and flattened single-batch layouts, so
canonicalization does not hide their existing fusion opportunities.

## Rule names and production boundaries

Upstream names in the tables are inventory identifiers, not the names emitted by
the target matcher. Target rule names describe the actual transformation:
`ComposeConsecutiveGathers`, `FuseConsecutiveConstantPads`,
`MoveCastBeforeTranspose`, `FactorCommonAddFromWhere`, and
`FuseSingleTokenLinearAttention`, for example. Parameterized variants identify
the operation, operand, or input form rather than appending Boolean flags.
Existing exported rule variables and aggregate rule sets remain stable.

The adopted subsets deliberately retain the following boundaries:

| Area | Required proof or restriction |
|---|---|
| Gather and partitions | Indices are in bounds before composition or conversion to Slice; partitions have distinct reads and cover the complete axis. Gather errors must not become Slice clamping. |
| Cast and layout movement | All conversion attributes are preserved. Moving across a shared layout must not duplicate conversion work. Gather is not moved before String/unknown-input Casts because that could hide conversion errors. |
| Padding | Constant, nonnegative pads, matching bitwise fill values (including BF16 and signed zero), valid axes, and nonoverflowing combined INT64 pads. |
| Expand movement | Known rank and broadcasting; moving computation before a potentially empty expansion is rejected when it could introduce evaluation such as integer division by zero. |
| Binary movement | Preserve concrete operator attributes, including `BitShift.direction` and `Mod.fmod`; do not move reference attributes whose values are not known. |
| Reshape | Valid one-dimensional targets, function-local `allowzero` semantics, and unambiguous element counts. Anonymous dimensions do not prove equality. |
| Range | Scalar constant signed-integer bounds and step, scalar/rank-one offset, nonzero step, and representable shifted bounds. Rank-increasing offsets are rejected. |
| Reduction to TopK | Integer types, matching axes and keepdims, and first-index ties only. Floating NaN/signed-zero ordering is not assumed equivalent. |
| Matrix operations | Static layout and broadcast proofs; expensive shared products are not duplicated. Floating unit-reduction MatMul is not replaced by Mul. |
| Rotary and attention compatibility | Positive even rotary dimensions, supported dtypes/opsets, and preserved rotary/attention attributes. Packed QKV fusion recognizes both Slice and Split partitions. |

ORT cos/sin cache conversion constructs entries for the positions actually used,
not every integer up to their maximum. Negative, sparse, and extreme INT64
positions therefore neither index the wrong row nor trigger huge optimizer-time
allocations. Distinct frequency inputs are not conflated, overridable frequency
initializers remain dynamic, and empty batches/sequences retain valid cache
shapes. Shape queries use proven equivalent dominating inputs when available so
they do not prevent subsequent partial-rotary or attention fusion.

## Regression coverage

The corresponding `*_test.py` modules check rule application as well as output
equivalence. Generated operator families are parameterized over their registered
operators and operand/layout variants, rather than assuming that one successful
`Add` or `Relu` case exercises every registration.

| Surface | Positive and safety coverage |
|---|---|
| Collections | Unary and scalar/broadcast binary movement, negative/scalar/vector Gather indices, partition arities 2–4 and permutations, reverse Slice strides, shared consumers and graph outputs, and preservation of out-of-range Gather errors. |
| Layout and Expand | Generated unary/binary/reduction variants, both operand sides and Where branches, operator attributes, broadcasting/rank proofs, empty dimensions, axes normalization, local opsets, and fixed-point behavior. |
| Canonicalization | All six neutral-vector variants, padding fills/axes/overflow, Reshape rank and `allowzero`, integer TopK ties and axes, Range dtype/rank/step bounds, and shared layout producers. |
| Matrix | Activation/layout variants with numerical comparisons, Gemm attributes/bias/sharing, reshape/broadcast constraints, integer-only unit reductions, GroupedMatMul index normalization/bias, and LinearAttention eligibility. |
| Boolean, shape, and labels | Broadcast and signed-zero/NaN preservation, dynamic axes, symbolic/anonymous dimensions, dominating reuse, overridable initializers, and string/integer mapping/default composition. |
| Core matcher | Rejected conditions, node and named/anonymous value checkers, fresh candidate bindings, distinct roots, shared internal producers, permutations, bound-input pruning, removability, and legacy single-result matcher compatibility. |
| Rotary and attention compatibility | Actual negative/sparse/extreme positions, FLOAT/FLOAT16 cache variants and Cast attributes, frequency expansion, empty caches, packed rank-two numerical equivalence, and partial/interleaved rotary with all eight MHA inputs and present KV outputs retained. |

Schema-dependent GroupedMatMul, LinearAttention, and standard RotaryEmbedding
cases are skipped only when the installed ONNX version lacks the required
schema or reference evaluator. Structural safety cases remain applicable to
older versions. The complete `onnxscript/rewriter` and `onnxscript/optimizer`
suites are the regression gate; branch instrumentation complements, rather
than replaces, numerical and nonapplication assertions.

## Canonicalization

Sources: [casts][casts], [clips][clips], [constants][constants],
[convolution][conv], [dropout][dropout], [identities][identity],
[initializer folding][initializer], [Not][not], [padding][pad], [STFT][stft].

| Upstream name | Disposition | Target or reason |
|---|---|---|
| `Cast` | Existing | `_basic_rules` and constant folding remove same-type casts. |
| `CastCast` | Existing, conservative subset | `_basic_rules` only allows recognized conversion chains; narrowing then widening is not an identity. |
| `CastCastBinary` | Not adopted | Changes arithmetic precision: promoted FP16 `2048 + 1` is `2049`, FP16 arithmetic is `2048`. |
| `CastOpCast` | Added, exact subset | `_canonical_optimizations` moves casts across data-preserving layout operations, not floating arithmetic. |
| `ClipClip` | Existing | `_fuse_relus_clips` merges successive clipping. |
| `ReluClipFusion` | Existing | `_fuse_relus_clips` checks effective clipping bounds. |
| `ConstantToInitializer` | Existing | Constant folding and `LiftConstantsToInitializersPass` already canonicalize storage. |
| `InitializerUnsqueezeCast` | Existing | Constant evaluation folds both operations before the nonconstant consumer. |
| `ConvBiasNull` | Existing | `_remove_optional_bias`. |
| `ConvAddFusion` | Not adopted as a new default | Floating bias folding changes rounding: `x=w=1`, bias `2^24`, then Add `-2^24` gives `0`; folded bias gives `1`. Existing combined affine rules remain opt-in. |
| `ConvMulFusion` | Not adopted as a new default | `x=w=1e20`, scale `1e-20` gives overflow before scaling, but finite output after weight folding. Existing combined affine rules remain opt-in. |
| `ConvBatchNormalizationFusion` | Existing | `_fuse_batchnorm`; no duplicate implementation or precision policy change. |
| `Dropout` | Existing | Constant folding handles inference/zero-ratio cases and optional masks. |
| `Identity` | Added, extended subset | Existing scalar/layout no-ops plus `_canonical_optimizations` for vector identities and zero-copy Reshape, and `_boolean_optimizations` for Boolean identities. Nonneutral epsilon and floating signed-zero changes are rejected. |
| `NotNot` | Added | `_boolean_optimizations`; intermediate Not can remain shared. |
| `PadConv` | Existing | `_fuse_pad_into_conv`. |
| `PadPadFusion` | Added | `_canonical_optimizations`; compatible constant fills and nonnegative padding only. |
| `STFTFusion` | Not adopted | Recognizing DFT weights within a tolerance and replacing convolution with a different floating evaluation is not exact, especially for NaN/Inf and rounding. |

## Collections

Sources: [Concat][concat], [Gather][gather], [Slice][slice], [Split][split],
[sequences][sequence], [shape queries][shape], [layer extraction][layer],
[partitioning][partition].

| Upstream name | Disposition | Target or reason |
|---|---|---|
| `ConcatEmpty` | Existing and extended | Constant folding already handles empty operands; `_collection_optimizations` also exposes this to direct rewriting. |
| `ConcatSliceElimination` | Added | `_collection_optimizations`; select the recovered operand without assuming all other consumers disappear. |
| `ConcatGather` | Added | `_collection_optimizations`; select one source and adjust constant indices. |
| `ConcatTwiceUnary` | Added | `_collection_optimizations`; factor compatible elementwise, Cast, and scalar-binary operations through concatenation. |
| `GatherConcat` | Added | `_collection_optimizations`; unified with constant-index Concat/Gather simplification. |
| `GatherGather` | Added | `_collection_optimizations`; compose indices without losing their rank. |
| `GatherShape` | Added | `_shape_optimizations`; bounded contiguous Shape slices, scalar rank restoration, opset 15+. |
| `PreShapeNodeElimination` | Added | `_shape_optimizations`; Shape bypasses Cast while preserving its bounds and other Cast uses. |
| `GathersSplit` | Added, fixed-arity subset | `_collection_optimizations`; complete two- to four-way static partitions. The matcher requires a fixed number of replacement outputs. |
| `GatherSliceToSplit` | Added, fixed-arity subset | `_collection_optimizations`; compatible complete two- or three-way static partitions. Higher output arities are not represented by these fixed-output patterns. |
| `GatherToSlice` | Added | `_collection_optimizations`; proven constant progressions, negative-index normalization, and scalar rank restoration. |
| `GatherUpstreamPropagation` | Added | `_collection_optimizations`; compatible producers only. |
| `SliceSlice` | Added | `_collection_optimizations`; disjoint axes with valid constant parameters. |
| `SliceElimination` | Existing | `_collapse_slices`; step, bounds, and shape checks. |
| `SlicesSplit` | Existing and extended | Existing two-way `_basic_rules` plus complete two- to four-way static partitions in `_collection_optimizations`. Higher arities are outside these fixed-output patterns. |
| `SplitConcat` | Added | `_collection_optimizations`; same axis, complete ordered outputs. |
| `SliceConcatToSpaceToDepth` | Added, restricted subset | `_collection_optimizations`; channel ordering must be equivalent. Phase-major multi-channel concatenation is not standard SpaceToDepth ordering. |
| `SequenceConstructAt` | Existing and extended | Constant folding already resolves sequence elements; direct rewriter coverage in `_collection_optimizations`. |
| `SplitToSequenceSequenceAt` | Existing | Constant folding lowers known sequence splits and resolves SequenceAt. |
| `ShapeTranspose` | Added | `_shape_optimizations`; permute the small shape vector rather than data, including default permutations and empty slices. |
| `UnsqueezeShape` | Added | `_shape_optimizations`; assemble source dimensions and inserted ones without touching data. |
| `RepeatedLayer` | Not adopted | Upstream opt-in function/region extraction, not reduced tensor computation. Default optimization inlines functions; execution-region policy belongs to a backend. |
| `PartitionGroup` | Not adopted | Upstream opt-in `ai.rt` region partitioning depends on transfer costs and an execution backend, not portable ONNX tensor optimization. |

## Expansion and selection

Sources: [expansion][expand] and [selection][where].

| Upstream name | Disposition | Target or reason |
|---|---|---|
| `NotWhere` | Added | `_boolean_optimizations`; exchange branches, preserving broadcasting and special values. |
| `UnsqueezeEqual` | Added | `_boolean_optimizations`; equal normalized axes and ranks, plus reuse of an already available sibling Unsqueeze. |
| `WhereAdd` | Added, exact subset | `_boolean_optimizations`; factor a common Add operand without changing its position. Multiplicative additive-mask conversions are not adopted because `0 * Inf` is NaN. |
| `Expand` | Existing | `_basic_rules` and constant folding. |
| `ExpandBroadcast` | Existing, enabled in default rewriting | `_remove_expand_before_binary_op`; check the actual resulting broadcast shape. |
| `ShapeBasedConcatExpand` | Added | `_shape_optimizations`; replace redundant dynamic target dimensions with unit broadcast dimensions, retaining shared shape expressions. |
| `ShapeBasedExpandBroadcast` | Existing, enabled in default rewriting | `_remove_expand_before_binary_op` includes symbolic output-shape proofs. |
| `ShapeBasedExpandBroadcastMatMul` | Added | `_matrix_optimizations`; prove both batch broadcasting and matrix dimensions. |
| `ShapeBasedStaticExpand` | Added | `_expand_optimizations`; materialize a proven output shape. |
| `ShapeBasedExpandSwap` | Added | `_expand_optimizations`; compute before broadcasting when shapes prove equivalence. |
| `ShapeBasedExpandCastWhereSwap` | Added | `_expand_optimizations`; retain exactly the same broadcast result. |
| `ExpandSwap` | Added | `_expand_optimizations`; eligible shape-preserving operators only. |
| `SwapExpandUnsqueeze` | Added | `_expand_optimizations`; remap inserted dimensions. |
| `ExpandUnsqueezeExpand` | Added | `_expand_optimizations`; combine redundant materialization. |
| `SwapExpandReshape` | Added | `_expand_optimizations`; only provably order-preserving cases. |

## Layout and reshape

Sources: [layout][layout], [reshape][reshape], [transpose][transpose],
[Unsqueeze][unsqueeze].

| Upstream name | Disposition | Target or reason |
|---|---|---|
| `ConcatReshape` | Existing, conservative subset | Constant folding and `_materialize_reshape_shape` materialize proven shapes; ambiguous multiple dynamic dimensions remain dynamic. |
| `Reshape` | Existing and extended | Constant folding, `_basic_rules`, and `_layout_optimizations`. |
| `ReduceReshape` | Added | `_layout_optimizations`; equivalent reduction axes/keepdims only. |
| `Reshape2Of3` | Added | `_layout_optimizations`; prove flattening and broadcasting equivalence. |
| `ReshapeReshapeBinary` | Added | `_layout_optimizations`; matching layouts and compatible pre-layout ranks. |
| `ReshapeReshape` | Existing | `_basic_rules`; preserves allowzero constraints. |
| `ReshapeSqueeze` | Added | `_layout_optimizations`; resolve singleton axes into the target shape. |
| `ShapeBasedEditDistanceReshape` | Existing, conservative subset | `_materialize_reshape_shape`; multiple unproven symbolic dimensions are not guessed. |
| `ShapeBasedReshapeIsSqueeze` | Added | `_layout_optimizations`; only inserted/removed unit dimensions. |
| `ShapedBasedReshape` | Added | `_layout_optimizations`; dimension copying with a proved unchanged result. |
| `StaticConcatReshape` | Existing | `_materialize_reshape_shape` replaces one inferred dimension when valid. |
| `UnsqueezeOrSqueezeReshape` | Added | `_layout_optimizations`; preserve allowzero and shape-inference semantics. |
| `UnsqueezeReshape` | Added | `_layout_optimizations`; compatible inserted dimensions. |
| `MulUnsqueezeUnsqueeze` | Added | `_layout_optimizations`; equal axes and broadcast-compatible input ranks. |
| `SqueezeAdd` | Added | `_layout_optimizations`; equal squeezes, with no arithmetic reassociation. |
| `SqueezeBinaryUnsqueeze` | Added | `_layout_optimizations`; scalar/rank-safe cancellation. |
| `SwapUnsqueezeTranspose` | Added | `_layout_optimizations`; valid normalized axes and remapped permutation. |
| `TransposeEqualReshape` | Added | `_layout_optimizations`; only unit-axis reordering. |
| `TransposeReshapeTranspose` | Added, restricted subset | `_layout_optimizations`; unit-dimension reshaping and composed permutations must prove less data movement. |
| `TransposeTranspose` | Existing | `_basic_rules` composes arbitrary valid explicit permutations. |
| `TransposeGather` | Added | `_layout_optimizations`; scalar index, corrected axis/permutation. |
| `UnsqueezeUnsqueeze` | Existing and extended | `_layout_optimizations` extends existing single-positive-axis handling to valid multi-axis/negative cases. |
| `SqueezeUnsqueeze` | Added | `_layout_optimizations`; equivalent axes, preserving unit dimensions. |

## Algebra and common subexpressions

Sources: [common rewrites][common], [multiplication][mul], [Range][range],
[reductions][reduce], [symbolic shape algebra][shape_algebra],
[subtraction][sub].

| Upstream name | Disposition | Target or reason |
|---|---|---|
| `DivMul` | Not adopted | Reciprocal rounding makes `(1/d)*x` differ from `x/d`; zero and infinite values also differ. |
| `MulMulMulScalar` | Not adopted | Reassociation changes rounding, overflow, NaN and signed-zero behavior. |
| `SwitchOrderBinary` | Not adopted | Floating Add/Mul reassociation is not exact. |
| `SwapRangeAddScalar` | Added, checked integer subset | `_canonical_optimizations`; constant bounds, step, offset and nonoverflowing shifted bounds. Floating/dynamic cases can change sequence length or values and are rejected. |
| `ReduceArgTopK` | Added, exact subset | `_canonical_optimizations`; compatible integer reductions and first-index tie handling. |
| `ReduceSumNormalize` | Not adopted | Moves reductions and arithmetic across floating precision boundaries. |
| `Sub1Mul` | Not adopted | Distribution changes floating rounding and exceptional values. |
| `SwapUnary` | Added, exact subset | `_canonical_optimizations` moves Cast across compatible layout operations; generic ordering without reduced work is not imposed. |
| `SameChildren` | Existing | CSE merges identical nodes and their descendant chains. |
| `SameChildrenFromInput` | Existing | CSE also handles graph-input consumers. |
| `ShapeBasedIdentity` | Existing | `_collapse_slices` checks shape, step, and effective start. |
| `ShapeBasedSameChildren` | Added | `_shape_optimizations`; share earlier Expand/Reshape results with the same proven shape, even when shape expressions differ. |
| `ShapeBasedShapeShapeAdd` | Not adopted | The inspected upstream class performs no rewrite. |

## Matrix, normalization, activation, and attention

Sources: [matrix operations][matmul], [normalization][normalization],
[activation][activation], [attention][attention].

| Upstream name | Disposition | Target or reason |
|---|---|---|
| `GroupedMatMul` | Added | `_matrix_optimizations`; canonical expert selection, valid indices and opset 29+. |
| `GemmTranspose` | Added | `_matrix_optimizations`; absorb rank-two input transposes into flags. |
| `GemmOutputTranspose` | Added | `_matrix_optimizations`; unbiased Gemm and rank-two output only. |
| `GemmSumFusion` | Added | `_matrix_optimizations`; exactly two Sum operands and a Gemm-compatible bias shape. |
| `MatMulAdd` | Existing | `_matmul_add_to_gemm`, explicitly opt-in. |
| `MatMulBatchNormalizationFusion` | Not adopted | Folding floating parameters changes arithmetic rounding/order, just like weight scaling. |
| `MatMulReshape2Of3` | Existing and extended | `_broadcast_to_matmul` plus `_matrix_optimizations`; remove batch-flattening Reshapes only after proving matrix layout and direct output shape. |
| `MatMulScaleFusion` | Not adopted | Moving scale into weights/alpha changes overflow, rounding, and special values. |
| `MulMulMatMul` | Not adopted | Moving two input scales past a reduction reassociates floating arithmetic. |
| `ReshapeMatMulReshape` | Existing | `_broadcast_to_matmul`. |
| `ShapeBasedMatMulToMul` | Added, integer subset | `_matrix_optimizations`; proved unit reduction dimension and valid broadcast result. Floating MatMul accumulation can turn a negative-zero product into positive zero. |
| `SwitchReshapeActivation` | Added | `_matrix_optimizations`; move elementwise activations before private layouts of matrix-product results to expose fusion, retaining operator attributes. |
| `TransposeMatMul` | Added | `_matrix_optimizations`; rank-two transpose absorption. |
| `TransposeReshapeMatMul` | Not adopted | Layout preference without a proved reduction in data movement is not imposed. |
| `BatchNormalization` | Not adopted as a new identity | Nonzero epsilon changes magnitude; even otherwise neutral parameters can change `-0` to `+0`. |
| `BatchNormalizationTraining` | Not adopted | Decomposition into primitives expands computation; it is a backend lowering rather than a portable optimization. |
| `CastLayerNormalizationCast` | Not adopted | Removing promotion changes rounding, including double-rounding cases. |
| `LayerNormalization` | Existing | `rules/fusion/_layer_norm`; no duplicate implementation. |
| `LayerNormalizationScale` | Not adopted | Folding a post-affine transform changes floating operation order. |
| `RMSNormalization` | Existing | `rules/fusion/_rms_normalization` and explicit ONNX/ORT fusion surfaces. |
| `RMSNormalizationMul` | Not adopted | Folding a post-scale changes floating rounding and overflow. |
| `Gelu` | Existing, opt-in | ORT Gelu fusions; approximate constants are not promoted to exact generic defaults. |
| `LeakyRelu` | Added, exact subset | `_matrix_optimizations`; finite nonnegative alpha exactly representable by the FLOAT attribute. Negative alpha changes signed zero: with `x=-0` and `alpha=-0.5`, the Greater-based decomposition gives `+0`, LeakyRelu gives `-0`. |
| `MaxRelu` | Added, ordered subset | `_matrix_optimizations`; only `Max(x, scalar +0)` with a Relu-compatible dtype/opset. Reversing the inputs or accepting negative zero is not equivalent. |
| `SoftmaxCrossEntropyLossCast` | Not adopted | Changes reduction precision and floating evaluation. |
| `RotaryEmbedding` | Existing | Explicit standard ONNX and ORT rotary fusion surfaces. |
| `RotaryConcatPart` | Existing, direct fusion | Target rotary decomposition matching avoids requiring the source's preparatory function canonicalization. |
| `FunctionCausalMask` | Not adopted | Encapsulation into a model-local function alone does not reduce runtime work. |
| `FunctionCausalMaskMulAdd` | Not adopted | Same function-encapsulation distinction; no generic runtime improvement. |
| `FunctionCosSinCache` | Existing, opt-in | ORT cache fusion already exists; source function encapsulation alone is not added. |
| `FunctionHalfRotaryEmbedding` | Existing | Direct rotary decomposition fusions. |
| `FunctionAttention` | Existing, opt-in | ORT attention/SDPA fusions; source function encapsulation is not a new default. |
| `LinearAttention` | Added, exact subset | `_matrix_optimizations`; FLOAT single-token linear recurrence with equal query/KV heads, explicit state, canonical packing, and finite nonzero scale, opset 27+. Zero scale is rejected because the operator interprets it as an automatic scale, not multiplication by zero; GQA broadcasting and different accumulation types are not conflated with this subset. |
| `FunctionAttentionGQA` | Existing, opt-in | ORT grouped-query attention fusion; no duplicate model-local wrapper. |
| `AttentionGQA` | Existing | `rules/fusion/_gqa` and the explicit ONNX/ORT fusion APIs. |

## Traditional machine learning

Sources: [LabelEncoder][label_encoder] and [TreeEnsemble][tree].

| Upstream name | Disposition | Target or reason |
|---|---|---|
| `LabelEncoderFusion` | Added | `_label_encoder`; compose distinct string/int64 list keys, values and defaults, preserving shared intermediates. Floating/NaN-key maps and legacy v1 encoders are not conflated with this subset. |
| `TreeEnsemble` | Not adopted | Conversion from classic classifiers/regressors to opset-5 TreeEnsemble is a representation/backend migration, not less tree computation. Classifiers additionally require label reconstruction nodes. |

[registry]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/dispatch_table.cc
[license]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/LICENSE
[notice]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/NOTICE
[casts]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/canonicalization/cast_pattern.h
[clips]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/canonicalization/clip_pattern.h
[constants]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/canonicalization/constant_pattern.h
[conv]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/canonicalization/conv_pattern.h
[dropout]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/canonicalization/dropout_pattern.h
[identity]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/canonicalization/identity_pattern.h
[initializer]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/canonicalization/initializer_unsqueeze_cast_pattern.h
[not]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/canonicalization/not_pattern.h
[pad]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/canonicalization/pad_pattern.h
[stft]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/canonicalization/stft_pattern.h
[concat]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/collections/concat_pattern.h
[gather]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/collections/gather_pattern.h
[slice]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/collections/slice_pattern.h
[split]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/collections/split_pattern.h
[sequence]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/collections/sequence_pattern.h
[shape]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/collections/shape_pattern.h
[layer]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/collections/layer_pattern.h
[partition]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/collections/partition_group_pattern.h
[expand]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/expand/expand_pattern.h
[where]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/expand/where_pattern.h
[layout]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/layout/layout_pattern.h
[reshape]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/reshape/reshape_pattern.h
[transpose]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/transpose/transpose_pattern.h
[unsqueeze]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/unsqueeze/unsqueeze_pattern.h
[common]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/algebra/common_pattern.h
[mul]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/algebra/mul_pattern.h
[range]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/algebra/range_pattern.h
[reduce]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/algebra/reduce_pattern.h
[shape_algebra]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/algebra/shape_pattern.h
[sub]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/algebra/sub_pattern.h
[matmul]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/matmul/matmul_pattern.h
[normalization]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/normalization/normalization_pattern.h
[activation]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/normalization/activation_pattern.h
[attention]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/attention/attention_pattern.h
[label_encoder]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/traditionalml/label_encoder_pattern.h
[tree]: https://github.com/xadupre/onnx-light/blob/43c2ac6aa6a393dfb7b90527638d0c1f22139c1e/onnx_light/onnx_extensions/patterns/traditionalml/tree_ensemble_pattern.h
