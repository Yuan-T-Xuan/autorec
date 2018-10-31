

hyper_params = [
    ("dim_of_latent_factor", [25, 50, 75, 100, 125, 150]),
    ("l2_reg", [0.1, 0.01, 0.001, 0.0001, 0.00001]),
    ("type_of_interaction", ["PairwiseEuDist", "PairwiseLog", "PointwiseMLPCE"]),
    ("eudist_margin", [0.5, 1.0, 1.5, 2.0]),
    ("mlp_dim1", [-1, 50, 75, 100, 125, 150]),
    ("mlp_dim2", [-1, 50, 75, 100, 125, 150]),
    ("mlp_dim3", [-1, 50, 75, 100, 125, 150])
]
