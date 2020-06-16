from deephyper.search.nas.model.space.op.gnn import GraphConv2, GlobalAvgPool2, ChebConv2, EdgeConditionedConv2,\
    GraphAttention2, GraphConvSkip2, ARMAConv2, APPNP2, GlobalAttentionPool2
from gnn_test2.qm9.load_data import load_data_qm9
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np

LAYER_K_, KWARGS_K_ = "layer", "kwargs"

TESTS = [{LAYER_K_: GraphConv2,
          KWARGS_K_: {"channels": 4, "activation": "relu"}},
         {LAYER_K_: ChebConv2,
          KWARGS_K_: {"channels": 4, "activation": "relu"}},
         {LAYER_K_: EdgeConditionedConv2,
          KWARGS_K_: {"channels": 4, "activation": "relu"}},
         {LAYER_K_: GraphAttention2,
          KWARGS_K_: {"channels": 4, "activation": "relu"}},
         {LAYER_K_: GraphConvSkip2,
          KWARGS_K_: {"channels": 4, "activation": "relu"}},
         {LAYER_K_: ARMAConv2,
          KWARGS_K_: {"channels": 4, "activation": "relu"}},
         {LAYER_K_: APPNP2,
          KWARGS_K_: {"channels": 4, "activation": "relu"}}
         ]


def _test_batch_mode(layer, **kwargs):
    ([X_train, A_train, E_train], y_train), ([X_test, A_test, E_test], y_test) = load_data_qm9()
    inputX = Input(shape=(8, 4,))
    inputA = Input(shape=(8, 8,))
    inputE = Input(shape=(8, 8, 3))
    if layer is EdgeConditionedConv2:
        gn = layer(**kwargs)([inputX, inputA, inputE])
        pl = GlobalAvgPool2()([gn])
        ds = Dense(1)(pl)
        model = Model(inputs=[inputX, inputA, inputE], outputs=ds)
        model.compile(loss="mse", optimizer="adam")
        model.fit([X_train, A_train, E_train], y_train,
                  validation_data=([X_test, A_test, E_test], y_test),
                  epochs=2)
    else:
        gn = layer(**kwargs)([inputX, inputA])
        pl = GlobalAttentionPool2()([gn])
        ds = Dense(1)(pl)
        model = Model(inputs=[inputX, inputA], outputs=ds)
        model.compile(loss="mse", optimizer="adam")
        model.fit([X_train, A_train], y_train,
                  validation_data=([X_test, A_test], y_test),
                  epochs=2)
    return


for id, test in enumerate(TESTS):
    _test_batch_mode(test[LAYER_K_], **test[KWARGS_K_])
    print()
