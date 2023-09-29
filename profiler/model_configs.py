# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

## "algo" stands for tensor parallel partition algorithm
model_prof_configs = {
    "resnet": {
        "dtype": "fp32",
        "model_size": ["500M", "2B", "4B", "6_8B", "13B"],
        "mbs": [16, 32, 48, 64],
        "algo": [0, 1]
    },
    "gpt": {
        "dtype": "fp16",
        "model_size": ["350M", "1_3B", "2_6B", "6_7B", "13B", "scale-layer"],
        "mbs": [1, 2, 4, 8],
        "algo": [0, 1]
    },
    "t5": {
        "dtype": "fp16",
        "model_size": ["770M", "3B", "6B", "11B"],
        "mbs": [1, 2, 4, 8],
        "algo": [0]
    }
}

# model_size: (num_layers, in_channels, width_factor, params_dtype)
resnet_configs = {
    "500M": ([3, 4, 6, 3], 224, 2, "fp32"),
    "1B": ([3, 4, 6, 3], 320, 2, "fp32"), 
    "2B": ([3, 4, 6, 3], 448, 2, "fp32"),
    "4B": ([3, 4, 6, 3], 640, 2, "fp32"),
    "6_8B": ([3, 4, 6, 3], 320, 16, "fp32"),
    "13B": ([3, 4, 23, 3], 320, 16, "fp32"),
}

# model_size: (num_layers, seq_len, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, params_dtype)
gpt_configs = {
    "350M": (1, 2048, 1024, 1024*4, 16, 1024//16, 51200, "fp16"),
    "1_3B": (1, 2048, 2048, 2048*4, 32, 2048//32, 51200, "fp16"),
    "2_6B": (1, 2048, 2560, 2560*4, 32, 2560//32, 51200, "fp16"),
    "6_7B": (1, 2048, 4096, 4096*4, 32, 4096//32, 51200, "fp16"),
    "13B": (1, 2048, 5120, 5120*4, 40, 5120//40, 51200, "fp16"),
    "scale-layer": (1, 2048, 512, 512*4, 8, 512//8, 51200, "fp16")
}

# model_size: (num_layers, encoder_seq_length, decoder_seq_length, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, params_dtype)
## T5-22B is obtained by doubling the layer number in T5-11B, thus share the same op-level profiling results
t5_configs = {
    "770M": (1, 2048, 512, 1024, 4096, 16, 64, 30592, "fp16"),
    "3B": (1, 2048, 512, 1024, 16384, 32, 128, 30592, "fp16"),
    "6B": (1, 2048, 512, 1024, 32768, 64, 128, 30592, "fp16"),
    "11B": (1, 2048, 512, 1024, 65536, 128, 128, 30592, "fp16"),    
}
