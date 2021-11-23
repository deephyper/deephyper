from deephyper.layers.padding import Padding

# When loading models with: "model.load('file.h5', custom_objects=custom_objects)"
custom_objects = {"Padding": Padding}

from deephyper.layers._mpnn import get_gcn_attention, get_mol_feature, get_all_mol_feat, \
    SPARSE_MPNN, MP_layer, Message_Passer_NNM, Update_Func_MLP, Update_Func_GRU, Attention_GAT, Attention_SYM_GAT, \
    Attention_GCN, Attention_Linear, Attention_Const, Attention_COS, Attention_Gen_Linear, GlobalAttentionPool, \
    GlobalAttentionSumPool, GlobalSumPool, GlobalMaxPool, GlobalAvgPool

__all__ = ["get_all_mol_feat", "get_mol_feature", "get_gcn_attention", "SPARSE_MPNN", "MP_layer", "Message_Passer_NNM",
           "Update_Func_GRU", "Update_Func_MLP", "Attention_COS", "Attention_Linear", "Attention_Const",
           "Attention_GAT", "Attention_GCN", "Attention_SYM_GAT", "Attention_Gen_Linear", "GlobalAttentionPool",
           "GlobalAttentionSumPool", "GlobalSumPool", "GlobalAvgPool", "GlobalMaxPool"]

