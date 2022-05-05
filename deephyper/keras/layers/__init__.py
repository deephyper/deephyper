from deephyper.keras.layers._mpnn import (
    AttentionConst,
    AttentionCOS,
    AttentionGAT,
    AttentionGCN,
    AttentionGenLinear,
    AttentionLinear,
    AttentionSymGAT,
    GlobalAttentionPool,
    GlobalAttentionSumPool,
    GlobalAvgPool,
    GlobalMaxPool,
    GlobalSumPool,
    MessagePasserNNM,
    MessagePassing,
    SparseMPNN,
    UpdateFuncGRU,
    UpdateFuncMLP,
)
from deephyper.keras.layers._padding import Padding

__all__ = [
    "AttentionConst",
    "AttentionCOS",
    "AttentionGAT",
    "AttentionGCN",
    "AttentionGenLinear",
    "AttentionLinear",
    "AttentionSymGAT",
    "GlobalAttentionPool",
    "GlobalAttentionSumPool",
    "GlobalAvgPool",
    "GlobalMaxPool",
    "GlobalSumPool",
    "MessagePasserNNM",
    "MessagePassing",
    "SparseMPNN",
    "UpdateFuncGRU",
    "UpdateFuncMLP",
]

# When loading models with: "model.load('file.h5', custom_objects=custom_objects)"
custom_objects = {"Padding": Padding}
