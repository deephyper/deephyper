# Deephyper GNN module

## Edit 1
```deephyper/deephyper/search/nas/model/space/op/gnn.py```

The code for all GNN building blocks.

### Edit 2
```deephyper/deephyper/search/nas/model/train_utils.py```

I added some self-defined metrics for experiments.

### Running an example code
```python
from deephyper.search.nas.model.space.op.gnn import SPARSE_MPNN, GlobalAvgPool
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
import numpy as np

X_node = np.random.random((100, 20, 8))  # 100 graphs each with 20 nodes and each node with 8 features
X_edge = np.random.random((100, 30, 6))  # each graph has 30 edges, each edge has 6 features
X_pair = np.random.randint(0, 20, (100, 30, 2))  # each edge is indexed with 2 integers (source node ID and targe node ID)
X_mask = np.ones((100, 20))  # each node has a mask (0 or 1) to indicate if an atom exists or is added by padding
X_degree = np.random.random((100, 30))  # each edge's inverse sqrt degree property which is used in constant attention
y = np.random.random((100, 3))  # the global property of the graph with 3 features

input_node = Input(shape=(20, 8))
input_pair = Input(shape=(30, 2))
input_edge = Input(shape=(30, 6))
input_mask = Input(shape=(20, ))
input_degree = Input(shape=(30, ))
l1 = SPARSE_MPNN(state_dim=5, 
                 T=3, 
                 aggr_method='mean', 
                 attn_method='sym-gat',
                 update_method='gru',
                 attn_head=2,
                 activation='relu')([input_node,
                                     input_pair,
                                     input_edge,
                                     input_mask, 
                                     input_degree])
l2 = GlobalAvgPool()(l1)
l3 = Flatten()(l2)
l4 = Dense(3)(l3)
model = Model(inputs=[input_node, input_pair, input_edge, input_mask, input_degree], 
              outputs=l4)
model.compile(loss='mse', optimizer='adam')
model.fit([X_node, X_pair, X_edge, X_mask, X_degree], y, epochs=20)
```
