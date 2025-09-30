# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import onnx
from onnxscript import ir
import onnx.helper
import numpy as np
import logging
import torch
import math

from transformers import AutoConfig
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

class Phi4MiniReasoningPostProcessor:
    def __init__(self, config: AutoConfig, io_dtype: ir.DataType = ir.DataType.FLOAT):
        self.config = config
        self.original_max_position_embeddings = getattr(config, "original_max_position_embeddings", 4096)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 131072)
            
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_size = self.hidden_size // self.num_attention_heads
        self.io_dtype: ir.DataType = ir.DataType(io_dtype)

        # Torch dtype mapping for ONNX IR DataType
        self.to_torch_dtype = {
            ir.DataType.FLOAT: torch.float32,
            ir.DataType.FLOAT16: torch.float16,
            ir.DataType.BFLOAT16: torch.bfloat16,
            ir.DataType.DOUBLE: torch.float64,
            ir.DataType.INT64: torch.int64,
            ir.DataType.INT32: torch.int32,
        }
        
        # Initialize rotary embedding attributes
        position_scale = getattr(config, "rope_position_scale", 1.0)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        rotemb_dim = int(self.head_size * partial_rotary_factor) if partial_rotary_factor != 1.0 else 0
        rope_theta = getattr(config, "rope_theta", getattr(config, "rope_embedding_base", 10000.0))
        
        self.rotemb_attrs = {
            "create_caches": True,                           # Create cos/sin caches for rotary embeddings
            "save_caches": True,                             # Auto-save cos/sin caches for rotary embeddings after creation
            "cache_length": self.max_position_embeddings,    # Cache length to use when creating cos/sin caches for rotary embeddings
            "theta": rope_theta,                             # Base value if calculating cos/sin caches from scratch
            "partial_rotary_factor": partial_rotary_factor,  # Factor for partial rotary embeddings
            "interleaved": 0,                                # Interleave the rotary embeddings (e.g. [0, 0, 0, 1, 1, 1] to [0, 1, 0, 1, 0, 1], RotaryEmbedding kernel expects a default value of 0)
            "rotary_embedding_dim": rotemb_dim,              # For partial rotary embeddings (RotaryEmbedding kernel expects a default value of 0)
            "rescale_factors": 1.0,                          # Rescale factors when calculating `inv_freq` in rotary embeddings
            "t_dtype": torch.int64,                          # Torch dtype when calculating `t` in rotary embeddings
            "position_scale": position_scale,                # Scale value when calculating `t` in rotary embeddings
            "mscale": 1.0,                                   # Magnitude scaling factor when scaling `emb.cos()/emb.sin()` in rotary embeddings
            "mscale_policy": "",                             # Magnitude scaling policy when scaling `emb.cos()/emb.sin()` in rotary embeddings
        }
        
        # Handle rope scaling configuration for multi-cache scenarios
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            if "short_factor" in config.rope_scaling:
                # For models with multiple rotary embedding caches (e.g. Phi-3 mini 128K)
                self.rotemb_attrs["mscale_policy"] = config.rope_scaling.get("type", "")
                short_factor = torch.tensor(config.rope_scaling["short_factor"], dtype=torch.float32)
                long_factor = torch.tensor(config.rope_scaling["long_factor"], dtype=torch.float32)

                short_mscale = config.rope_scaling.get("short_mscale", 0)
                long_mscale = config.rope_scaling.get("long_mscale", 0)
                short_mscale = short_mscale if short_mscale > 0 else self.make_mscale(self.max_position_embeddings / self.original_max_position_embeddings)
                long_mscale = long_mscale if long_mscale > 0 else self.make_mscale(self.max_position_embeddings / self.original_max_position_embeddings)

                self.rotemb_attrs["multi_cache"] = {
                    "short_factor": short_factor,                # Short factor when calculating `inv_freq` in rotary embeddings
                    "long_factor": long_factor,                  # Long factor when calculating `inv_freq` in rotary embeddings
                    "short_mscale": short_mscale,                # Magnitude scaling for short factor when scaling `emb.cos()/emb.sin()` in rotary embeddings
                    "long_mscale": long_mscale,                  # Magnitude scaling for long factor when scaling `emb.cos()/emb.sin()` in rotary embeddings
                }

    @dataclass
    class PatternNodes:
        """Container for the nodes found in the old Cos/Sin value generation pattern."""
        gather_value: Optional[ir.Value] = None
        matmul_node: Optional[ir.Node] = None
        cos_node: Optional[ir.Node] = None
        sin_node: Optional[ir.Node] = None

    @dataclass
    class CacheData:
        """Container for generated cache data."""
        cos_large: np.ndarray
        sin_large: np.ndarray
        cos_small: np.ndarray
        sin_small: np.ndarray

    @dataclass
    class IfNodeComponents:
        """Container for If node components."""
        threshold_const_node: ir.Node
        greater_node: ir.Node
        if_node: ir.Node
        cos_output: ir.Value
        sin_output: ir.Value

    @dataclass
    class ProcessingChainNodes:
        """Container for position processing chain nodes."""
        position_ids_input: Optional[ir.Value] = None
        reduce_max_node: Optional[ir.Node] = None
        add_node: Optional[ir.Node] = None
        range_node: Optional[ir.Node] = None
        reshape_node: Optional[ir.Node] = None
        cast_node: Optional[ir.Node] = None
        constant_nodes: List[ir.Node] = field(default_factory=list)

    def make_mscale(self, mscale: float) -> float:
        """Calculate magnitude scaling factor for RoPE."""
        if mscale <= 1.0:
            return 1.0
        return math.sqrt(1 + math.log(mscale) / math.log(self.original_max_position_embeddings))

    def calculate_rotary_embedding_caches(self):
        """Generate cos/sin caches from scratch using the current rotemb_attrs."""
        if self.rotemb_attrs["rotary_embedding_dim"] > 0:
            dim = self.rotemb_attrs["rotary_embedding_dim"]
        else:
            dim = int(self.rotemb_attrs["partial_rotary_factor"] * self.head_size)
        
        inv_freq, attention_factor = self._compute_longrope_parameters(
            cache_length=self.rotemb_attrs["cache_length"],
            dim=dim
        )
        
        cache_length = self.rotemb_attrs["cache_length"]
        position_ids = torch.arange(cache_length, dtype=torch.int64).unsqueeze(0)  # Shape: (1, cache_length)
        
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)  # (1, dim//2, 1)
        position_ids_expanded = position_ids[:, None, :].float()  # (1, 1, cache_length)
        
        device_type = "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)  # (1, cache_length, dim//2)
            emb = torch.cat((freqs, freqs), dim=-1)  # (1, cache_length, dim)
            cos_cache = emb.cos() * attention_factor  # (1, cache_length, dim)
            sin_cache = emb.sin() * attention_factor  # (1, cache_length, dim)
        
        return cos_cache, sin_cache

    def _compute_longrope_parameters(self, cache_length: int, dim: int) -> tuple:
        """
        Computes the inverse frequencies with LongRoPE scaling for Phi-4.
        Based on the official transformers implementation.
        """
        base = self.rotemb_attrs["theta"]

        # Check if we have multi_cache configuration (LongRoPE)
        if "multi_cache" in self.rotemb_attrs:
            long_factor = self.rotemb_attrs["multi_cache"]["long_factor"]
            short_factor = self.rotemb_attrs["multi_cache"]["short_factor"]
            
            # Select factor based on cache length vs original max position embeddings
            if cache_length > self.original_max_position_embeddings:
                ext_factors = torch.tensor(long_factor, dtype=torch.float32, device="cpu")
                attention_factor = self.rotemb_attrs["multi_cache"]["long_mscale"]
            else:
                ext_factors = torch.tensor(short_factor, dtype=torch.float32, device="cpu")
                attention_factor = self.rotemb_attrs["multi_cache"]["short_mscale"]
        
        inv_freq_shape = torch.arange(0, dim, 2, dtype=torch.int64, device="cpu").float() / dim
        inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)
        
        if "rescale_inv_freq" in self.rotemb_attrs:
            inv_freq = self.make_inv_freq_rescaled(inv_freq)
        
        return inv_freq, attention_factor

    def reformat_rotary_embedding_caches(self):
        """Generate and format cos/sin caches for the current configuration."""
        cos_cache, sin_cache = self.calculate_rotary_embedding_caches()

        # Convert to the target dtype
        cos_cache = cos_cache.to(self.to_torch_dtype[self.io_dtype])
        sin_cache = sin_cache.to(self.to_torch_dtype[self.io_dtype])

        # Slice cos/sin caches from (M, H) to (M, H/2)
        hidden_dim = cos_cache.shape[-1]
        cos_cache = cos_cache.squeeze()[:, : (hidden_dim // 2)]
        cos_cache = cos_cache.to(self.to_torch_dtype[self.io_dtype])
        sin_cache = sin_cache.squeeze()[:, : (hidden_dim // 2)]
        sin_cache = sin_cache.to(self.to_torch_dtype[self.io_dtype])

        # Slice cos/sin caches from (M, H/2) to (M, R/2) if partial rotary embeddings are used
        if self.rotemb_attrs["partial_rotary_factor"] != 1.0:
            cos_cache = cos_cache[:, : (self.rotemb_attrs["rotary_embedding_dim"] // 2)]
            sin_cache = sin_cache[:, : (self.rotemb_attrs["rotary_embedding_dim"] // 2)]

        return cos_cache, sin_cache

    def make_inv_freq_rescaled(self, inv_freq):
        scale_factor = self.rotemb_attrs["rescale_inv_freq"]["factor"]
        low_freq_factor = self.rotemb_attrs["rescale_inv_freq"]["low_freq_factor"]
        high_freq_factor = self.rotemb_attrs["rescale_inv_freq"]["high_freq_factor"]
        old_context_len = self.original_max_position_embeddings

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []
        for freq in inv_freq:
            wavelen = 2 * torch.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)

        return torch.tensor(new_freqs, dtype=inv_freq.dtype)
    
    def delete_position_processing_nodes(self, model: ir.Model) -> ir.Model:
        """
        Delete the position processing nodes from the ONNX IR graph.
        This removes the sequence: position_ids -> ReduceMax -> Add -> Range -> Reshape -> Cast
        
        Args:
            model: ONNX IR Model to modify
            
        Returns:
            Modified ONNX IR Model with nodes removed
        """
        graph = model.graph
        
        # Step 1: Find position processing chain nodes
        chain_nodes = self._find_position_processing_chain(graph)
        if not self._validate_processing_chain(chain_nodes):
            return model
        
        # Step 2: Find constants that feed the chain
        self._find_chain_feeding_constants(graph, chain_nodes)
        
        # Step 3: Remove the processing chain nodes
        self._remove_processing_chain_nodes(graph, chain_nodes)
        
        # Step 4: Clean up position_ids input if unused
        self._cleanup_position_ids_input(graph, chain_nodes.position_ids_input)
        
        return model

    def _find_position_processing_chain(self, graph) -> ProcessingChainNodes:
        """Find the position processing chain nodes in the graph."""
        chain = self.ProcessingChainNodes()
        
        # Find position_ids input
        chain.position_ids_input = self._find_position_ids_input(graph)
        if not chain.position_ids_input:
            return chain
        
        # Find processing nodes in sequence
        chain.reduce_max_node = self._find_reduce_max_node(graph, chain.position_ids_input)
        
        if chain.reduce_max_node:
            chain.add_node = self._find_add_node(graph, chain.reduce_max_node)
        
        if chain.add_node:
            chain.range_node = self._find_range_node(graph, chain.add_node)
        
        if chain.range_node:
            chain.reshape_node = self._find_reshape_node(graph, chain.range_node)
        
        if chain.reshape_node:
            chain.cast_node = self._find_cast_node(graph, chain.reshape_node)
        
        return chain

    def _find_position_ids_input(self, graph) -> Optional[ir.Value]:
        """Find the position_ids input in the graph."""
        for input_val in graph.inputs:
            if "position_ids" in input_val.name:
                logging.info(f"Found position_ids input: {input_val.name}")
                return input_val
        
        logging.warning("position_ids input not found")
        return None

    def _find_reduce_max_node(self, graph, position_ids_input: ir.Value) -> Optional[ir.Node]:
        """Find ReduceMax node that processes position_ids."""
        for node in graph:
            if node.op_type == "ReduceMax":
                if any(input_val == position_ids_input for input_val in node.inputs):
                    logging.info(f"Found ReduceMax node: {node.name}")
                    return node
        return None

    def _find_add_node(self, graph, reduce_max_node: ir.Node) -> Optional[ir.Node]:
        """Find Add node that follows ReduceMax."""
        reduce_max_outputs = reduce_max_node.outputs
        for node in graph:
            if node.op_type == "Add":
                if any(input_val in reduce_max_outputs for input_val in node.inputs):
                    logging.info(f"Found Add node following ReduceMax: {node.name}")
                    return node
        return None

    def _find_range_node(self, graph, add_node: ir.Node) -> Optional[ir.Node]:
        """Find Range node that follows Add."""
        add_outputs = add_node.outputs
        for node in graph:
            if node.op_type == "Range":
                if any(input_val in add_outputs for input_val in node.inputs):
                    logging.info(f"Found Range node following Add: {node.name}")
                    return node
        return None

    def _find_reshape_node(self, graph, range_node: ir.Node) -> Optional[ir.Node]:
        """Find Reshape node that follows Range."""
        range_outputs = range_node.outputs
        for node in graph:
            if node.op_type == "Reshape":
                if any(input_val in range_outputs for input_val in node.inputs):
                    logging.info(f"Found Reshape node following Range: {node.name}")
                    return node
        return None

    def _find_cast_node(self, graph, reshape_node: ir.Node) -> Optional[ir.Node]:
        """Find Cast node that follows Reshape."""
        reshape_outputs = reshape_node.outputs
        for node in graph:
            if node.op_type == "Cast":
                if any(input_val in reshape_outputs for input_val in node.inputs):
                    logging.info(f"Found Cast node following Reshape: {node.name}")
                    return node
        return None

    def _validate_processing_chain(self, chain_nodes: ProcessingChainNodes) -> bool:
        """Validate that sufficient chain nodes were found for deletion."""
        if not chain_nodes.position_ids_input:
            logging.warning("Cannot delete processing chain: position_ids input not found")
            return False
        
        # We need at least the reduce_max_node to proceed
        if not chain_nodes.reduce_max_node:
            logging.warning("Cannot delete processing chain: ReduceMax node not found")
            return False
        
        # Log found nodes
        found_nodes = []
        if chain_nodes.reduce_max_node:
            found_nodes.append(f"ReduceMax: {chain_nodes.reduce_max_node.name}")
        if chain_nodes.add_node:
            found_nodes.append(f"Add: {chain_nodes.add_node.name}")
        if chain_nodes.range_node:
            found_nodes.append(f"Range: {chain_nodes.range_node.name}")
        if chain_nodes.reshape_node:
            found_nodes.append(f"Reshape: {chain_nodes.reshape_node.name}")
        if chain_nodes.cast_node:
            found_nodes.append(f"Cast: {chain_nodes.cast_node.name}")
        
        logging.info(f"Found position processing chain: {', '.join(found_nodes)}")
        return True

    def _find_chain_feeding_constants(self, graph, chain_nodes: ProcessingChainNodes) -> None:
        """Find constant nodes that exclusively feed the processing chain."""
        chain_node_list = [
            node for node in [
                chain_nodes.reduce_max_node,
                chain_nodes.add_node,
                chain_nodes.range_node,
                chain_nodes.reshape_node,
                chain_nodes.cast_node
            ] if node is not None
        ]
        
        for node in graph:
            if node.op_type == "Constant":
                constant_output = node.outputs[0] if node.outputs else None
                if constant_output and self._constant_feeds_chain_exclusively(
                    graph, constant_output, chain_node_list, node
                ):
                    chain_nodes.constant_nodes.append(node)
                    logging.info(f"Found constant node feeding chain: {node.name}")

    def _constant_feeds_chain_exclusively(
        self, 
        graph, 
        constant_output: ir.Value, 
        chain_nodes: List[ir.Node], 
        constant_node: ir.Node
    ) -> bool:
        """Check if a constant exclusively feeds the processing chain."""
        # Check if constant feeds any chain node
        feeds_chain = any(
            any(input_val == constant_output for input_val in chain_node.inputs)
            for chain_node in chain_nodes
        )
        
        if not feeds_chain:
            return False
        
        # Check if constant is used by any non-chain nodes
        for node in graph:
            if node not in chain_nodes and node != constant_node:
                if any(input_val == constant_output for input_val in node.inputs):
                    return False
        
        return True

    def _remove_processing_chain_nodes(self, graph, chain_nodes: ProcessingChainNodes) -> None:
        """Remove all processing chain nodes from the graph."""
        nodes_to_delete = [
            node for node in [
                chain_nodes.reduce_max_node,
                chain_nodes.add_node,
                chain_nodes.range_node,
                chain_nodes.reshape_node,
                chain_nodes.cast_node
            ] if node is not None
        ]
        nodes_to_delete.extend(chain_nodes.constant_nodes)
        
        if nodes_to_delete:
            self._delete_nodes_from_graph(graph, nodes_to_delete)
        else:
            logging.warning("No processing chain nodes found to delete")

    def _delete_nodes_from_graph(self, graph, nodes_to_delete: List[ir.Node]) -> None:
        """Delete nodes from the graph with error handling."""
        try:
            graph.remove(nodes_to_delete)
            logging.info(f"Successfully deleted {len(nodes_to_delete)} processing chain nodes")
        except Exception as e:
            logging.error(f"Error deleting nodes in batch: {e}")
            # Try deleting nodes one by one
            self._delete_nodes_individually(graph, nodes_to_delete)
    """ 
    def _delete_nodes_individually(self, graph, nodes_to_delete: List[ir.Node]) -> None:
        Delete nodes individually with error handling.
        for node in nodes_to_delete:
            try:
                graph.remove([node])
                logging.info(f"Successfully deleted node: {node.name}")
            except Exception as e:
                logging.error(f"Failed to delete node {node.name}: {e}")
    """
    def _cleanup_position_ids_input(self, graph, position_ids_input: Optional[ir.Value]) -> None:
        """Remove position_ids input if it's no longer used."""
        if not position_ids_input:
            return
        
        # Check if position_ids is still used by any remaining nodes
        if self._input_still_used(graph, position_ids_input):
            logging.info(f"position_ids input {position_ids_input.name} is still in use")
            return
        
        try:
            graph.inputs.remove(position_ids_input)
            logging.info(f"Removed unused position_ids input: {position_ids_input.name}")
        except Exception as e:
            logging.warning(f"Could not remove position_ids input: {e}")

    def _input_still_used(self, graph, input_value: ir.Value) -> bool:
        """Check if an input value is still used by any nodes in the graph."""
        return any(
            any(input_val == input_value for input_val in node.inputs)
            for node in graph
        )

    def insert_rotary_embedding_caches(self, model: ir.Model, threshold: int = 4096) -> ir.Model:
        """
        Replaces the current Cos/Sin value generation with an control flow node containing 
        cached Cos/Sin values.
        
        Args:
            model: ONNX IR Model to modify
            threshold: Threshold value for Phi-4-mini-reasoning cache selection (default: 4096)
            
        Returns:
            Modified ONNX IR Model with MatMul→Cos/Sin replaced by cache-enabled If node
        """
        graph = model.graph
        
        # Step 1: Find pattern nodes
        pattern = self._find_pattern_nodes(graph)
        if not self._validate_pattern_nodes(pattern):
            return model
        
        # Step 2: Generate cache data
        cache_data = self._generate_cache_data()
        
        # Step 3: Create If node with caches
        if_components = self._create_if_node_with_caches(cache_data, threshold, pattern.gather_value)
        
        # Step 4: Replace pattern with If node
        self._replace_pattern_with_if_node(graph, pattern, if_components)
        
        # Step 5: Clean up old nodes
        self._remove_old_nodes(graph, pattern)
        
        return model


    def _find_pattern_nodes(self, graph) -> PatternNodes:
        """Find the MatMul→Cos/Sin pattern nodes in the graph."""
        pattern = self.PatternNodes()
        
        # Find attention mask gather chain
        pattern.gather_value = self._find_attention_mask_gather_value(graph)
        
        # Find MatMul→Cos/Sin pattern
        matmul_cos_sin = self._find_matmul_cos_sin_nodes(graph)
        pattern.matmul_node = matmul_cos_sin[0]
        pattern.cos_node = matmul_cos_sin[1]
        pattern.sin_node = matmul_cos_sin[2]
        
        return pattern

    def _find_attention_mask_gather_value(self, graph) -> Optional[ir.Value]:
        """
        Find the gather value from the attention mask processing chain.
        Chain: attention_mask → Shape → Gather
        """
        ATTENTION_MASK_NAME = "attention_mask"
        
        # Find Shape node that processes attention_mask
        shape_output_name = None
        for node in graph:
            if node.op_type == "Shape":
                for input_value in node.inputs:
                    if ATTENTION_MASK_NAME in input_value.name:
                        shape_output_name = node.outputs[0].name if node.outputs else None
                        break
                if shape_output_name:
                    break
        
        if not shape_output_name:
            return None
        
        # Find Gather node that follows the Shape
        for node in graph:
            if node.op_type == "Gather":
                for input_value in node.inputs:
                    if input_value.name == shape_output_name:
                        return node.outputs[0] if node.outputs else None
        
        return None

    def _find_matmul_cos_sin_nodes(self, graph) -> Tuple[Optional[ir.Node], Optional[ir.Node], Optional[ir.Node]]:
        """
        Find MatMul node that feeds into both Cos and Sin nodes.
        
        Returns:
            Tuple of (matmul_node, cos_node, sin_node)
        """
        for node in graph:
            if node.op_type == "MatMul":
                matmul_output = node.outputs[0] if node.outputs else None
                if matmul_output:
                    cos_node, sin_node = self._find_cos_sin_consumers(graph, matmul_output)
                    
                    if cos_node and sin_node:
                        logging.info(f"Found target MatMul node '{node.name}' that feeds into Cos and Sin nodes")
                        return node, cos_node, sin_node
        
        return None, None, None

    def _find_cos_sin_consumers(self, graph, matmul_output: ir.Value) -> Tuple[Optional[ir.Node], Optional[ir.Node]]:
        """Find Cos and Sin nodes that consume the MatMul output."""
        cos_node = None
        sin_node = None
        
        for consumer_node in graph:
            if consumer_node.op_type == "Cos":
                if self._node_consumes_value(consumer_node, matmul_output):
                    cos_node = consumer_node
            elif consumer_node.op_type == "Sin":
                if self._node_consumes_value(consumer_node, matmul_output):
                    sin_node = consumer_node
        
        return cos_node, sin_node

    def _node_consumes_value(self, node: ir.Node, value: ir.Value) -> bool:
        """Check if a node consumes the given value as input."""
        return any(input_val == value for input_val in node.inputs)

    def _validate_pattern_nodes(self, pattern: PatternNodes) -> bool:
        """Validate that all required pattern nodes were found."""
        if not pattern.gather_value:
            logging.warning("Error: Could not find attention mask gather node")
            return False
        
        if not pattern.matmul_node:
            logging.warning("Error: Could not find MatMul node that feeds into Cos and Sin nodes")
            return False
        
        if not pattern.cos_node or not pattern.sin_node:
            logging.warning("Error: Could not find both Cos and Sin nodes fed by the MatMul")
            return False
        
        # Log found pattern
        logging.info(f"Found MatMul→Cos/Sin pattern:")
        logging.info(f"MatMul: {pattern.matmul_node.name}")
        logging.info(f"Cos: {pattern.cos_node.name}")
        logging.info(f"Sin: {pattern.sin_node.name}")

        return True

    def _generate_cache_data(self) -> CacheData:
        """Generate cos/sin cache data for both large and small scenarios."""
        original_cache_length = self.rotemb_attrs["cache_length"]
        
        try:
            # Generate large cache (for long sequences)
            self.rotemb_attrs["cache_length"] = self.max_position_embeddings
            if "multi_cache" in self.rotemb_attrs:
                self.rotemb_attrs["rescale_factors"] = self.rotemb_attrs["multi_cache"]["long_factor"]
                self.rotemb_attrs["mscale"] = self.rotemb_attrs["multi_cache"]["long_mscale"]
            cos_cache_large, sin_cache_large = self.reformat_rotary_embedding_caches()
            
            # Generate small cache (for short sequences)
            self.rotemb_attrs["cache_length"] = self.original_max_position_embeddings
            if "multi_cache" in self.rotemb_attrs:
                self.rotemb_attrs["rescale_factors"] = self.rotemb_attrs["multi_cache"]["short_factor"]
                self.rotemb_attrs["mscale"] = self.rotemb_attrs["multi_cache"]["short_mscale"]
            cos_cache_small, sin_cache_small = self.reformat_rotary_embedding_caches()
            
            # Convert to numpy arrays for ONNX
            cache_data = self.CacheData(
                cos_large=cos_cache_large.detach().cpu().numpy(),
                sin_large=sin_cache_large.detach().cpu().numpy(),
                cos_small=cos_cache_small.detach().cpu().numpy(),
                sin_small=sin_cache_small.detach().cpu().numpy()
            )
            
            logging.info(f"Generated caches - Large: {cache_data.cos_large.shape}, Small: {cache_data.cos_small.shape}")
            return cache_data
            
        finally:
            # Restore original cache length
            self.rotemb_attrs["cache_length"] = original_cache_length

    def _create_if_node_with_caches(self, cache_data: CacheData, threshold: int, gather_value: ir.Value) -> IfNodeComponents:
        """Create the If node with cache branches."""
        # Create threshold comparison
        threshold_const_node, greater_node = self._create_threshold_comparison(threshold, gather_value)
        
        # Create cache branches
        then_branch = self._create_cache_branch(cache_data.cos_large, cache_data.sin_large, "large")
        else_branch = self._create_cache_branch(cache_data.cos_small, cache_data.sin_small, "small")
        
        # Create If node outputs
        if_cos_output = ir.Value(
            name="cos_cache",
            type=ir.TensorType(self.io_dtype),
            shape=ir.Shape(["max_sequence_length", "head_dim / 2"])
        )
        
        if_sin_output = ir.Value(
            name="sin_cache", 
            type=ir.TensorType(self.io_dtype),
            shape=ir.Shape(["max_sequence_length", "head_dim / 2"])
        )
        
        # Create the If node
        if_node = ir.node(
            "If",
            inputs=[greater_node.outputs[0]],
            outputs=[if_cos_output, if_sin_output],
            name="cos_sin_cache_if",
            attributes={
                "then_branch": ir.Attr("then_branch", ir.AttributeType.GRAPH, then_branch),
                "else_branch": ir.Attr("else_branch", ir.AttributeType.GRAPH, else_branch)
            }
        )
        
        return self.IfNodeComponents(
            threshold_const_node=threshold_const_node,
            greater_node=greater_node,
            if_node=if_node,
            cos_output=if_cos_output,
            sin_output=if_sin_output
        )

    def _create_threshold_comparison(self, threshold: int, gather_value: ir.Value) -> Tuple[ir.Node, ir.Node]:
        """Create threshold constant and greater comparison nodes."""
        # Create threshold constant
        threshold_const_name = f"threshold_const_{threshold}"
        threshold_value = ir.Value(
            name=threshold_const_name,
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape([])
        )
        threshold_value.const_value = ir.tensor(threshold, dtype=ir.DataType.INT64)
        
        threshold_const_node = ir.node(
            "Constant",
            inputs=[],
            outputs=[threshold_value],
            name=f"Constant_{threshold}",
            attributes={"value": ir.tensor(threshold, dtype=ir.DataType.INT64)}
        )
        
        # Create Greater node
        greater_output_value = ir.Value(
            name=f"greater_output_{threshold}",
            type=ir.TensorType(ir.DataType.BOOL),
            shape=ir.Shape([])
        )
        
        greater_node = ir.node(
            "Greater",
            inputs=[gather_value, threshold_value],
            outputs=[greater_output_value],
            name=f"Greater_{threshold}"
        )
        
        return threshold_const_node, greater_node

    def _create_cache_branch(self, cos_cache: np.ndarray, sin_cache: np.ndarray, branch_type: str) -> ir.Graph:
        """Create a cache branch for the If node."""
        # Create cache constant values and nodes
        cos_cache_value = ir.Value(
            name=f"cos_cache_{branch_type}", 
            type=ir.TensorType(self.io_dtype), 
            shape=ir.Shape(cos_cache.shape)
        )
        cos_cache_node = ir.node(
            "Constant", 
            inputs=[], 
            outputs=[cos_cache_value],
            name=f"{branch_type}_cos_cache_Constant", 
            attributes={"value": ir.tensor(cos_cache, dtype=self.io_dtype)}
        )
        
        sin_cache_value = ir.Value(
            name=f"sin_cache_{branch_type}", 
            type=ir.TensorType(self.io_dtype), 
            shape=ir.Shape(sin_cache.shape)
        )
        sin_cache_node = ir.node(
            "Constant", 
            inputs=[], 
            outputs=[sin_cache_value],
            name=f"{branch_type}_sin_cache_Constant", 
            attributes={"value": ir.tensor(sin_cache, dtype=self.io_dtype)}
        )
        
        # Create subgraph
        return ir.Graph(
            inputs=[],
            outputs=[cos_cache_value, sin_cache_value],
            nodes=[cos_cache_node, sin_cache_node],
            name=f"{branch_type}_rotemb_caches_graph",
        )

    def _replace_pattern_with_if_node(self, graph, pattern: PatternNodes, if_components: IfNodeComponents) -> None:
        """Replace the pattern nodes with the If node."""
        # Find all consumers of the original Cos and Sin outputs
        cos_consumers = self._find_value_consumers(graph, pattern.cos_node.outputs[0])
        sin_consumers = self._find_value_consumers(graph, pattern.sin_node.outputs[0])
        
        # Replace references to original outputs with If node outputs
        self._update_consumers(cos_consumers, if_components.cos_output)
        self._update_consumers(sin_consumers, if_components.sin_output)
        
        # Update GroupQueryAttention nodes if present
        self._update_group_query_attention_nodes(graph, if_components)
        
        # Add new nodes to the graph
        graph.append(if_components.threshold_const_node)
        graph.append(if_components.greater_node)
        graph.append(if_components.if_node)

    def _find_value_consumers(self, graph, value: ir.Value) -> List[Tuple[ir.Node, int]]:
        """Find all nodes that consume a given value."""
        consumers = []
        for node in graph:
            for i, input_val in enumerate(node.inputs):
                if input_val == value:
                    consumers.append((node, i))
        return consumers

    def _update_consumers(self, consumers: List[Tuple[ir.Node, int]], new_value: ir.Value) -> None:
        """Update consumer nodes to use a new value."""
        for node, input_idx in consumers:
            try:
                ir.Node.replace_input_with(node, input_idx, new_value)
            except Exception as e:
                logging.warning(f"Warning: Could not update {node.name or 'unnamed_node'} input[{input_idx}]: {e}")

    def _update_group_query_attention_nodes(self, graph, if_components: IfNodeComponents) -> None:
        """Update GroupQueryAttention nodes to use cache inputs."""
        gqa_nodes = [node for node in graph if node.op_type == "GroupQueryAttention"]
        
        for gqa_node in gqa_nodes:
            node_name = gqa_node.name or "GroupQueryAttention_node"
            try:
                # Replace cos_cache at position 7 and sin_cache at position 8
                if len(gqa_node.inputs) > 7:
                    ir.Node.replace_input_with(gqa_node, 7, if_components.cos_output)
                
                if len(gqa_node.inputs) > 8:
                    ir.Node.replace_input_with(gqa_node, 8, if_components.sin_output)
                    
            except Exception as e:
                logging.warning(f"Warning: Could not update {node_name} inputs: {e}")

    def _remove_old_nodes(self, graph, pattern: PatternNodes) -> None:
        """Remove the old MatMul, Cos, and Sin nodes."""
        nodes_to_remove = [pattern.matmul_node, pattern.cos_node, pattern.sin_node]
        
        try:
            graph.remove(nodes_to_remove)
            logging.info(f"Successfully removed MatMul→Cos/Sin sequence")
        except Exception as e:
            logging.warning(f"Warning: Could not remove some nodes: {e}")
            # Try removing nodes one by one
            for node in nodes_to_remove:
                try:
                    graph.remove([node])
                    logging.info(f"Removed {node.op_type} node: {node.name}")
                except Exception as e2:
                    logging.warning(f"Could not remove {node.op_type} node {node.name}: {e2}")
