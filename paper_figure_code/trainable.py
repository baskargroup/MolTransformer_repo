from MolTransformer import *
# Example 1: Initialize a self-supervised model with default settings



build_model_instance2 = BuildModel(model_mode='multiF_HF',dataset = 'ocelot', pretrain_model_file= '/Users/tcpba/MolTransformer_repo/MolTransformer/model/models/best_models/MultiF_HF/ocelot_aea/model_noneDDP.pt')
model2 = build_model_instance2.model
print("Loaded multiF_HF model")
trainable_params2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
print(f"Number of trainable parameters 2 in 'multiF_HF' mode: {trainable_params2}")

build_model_instance3 = BuildModel(model_mode='SS')
model3 = build_model_instance3.model
print("Loaded SS model")
trainable_params3 = sum(p.numel() for p in model3.parameters() if p.requires_grad)
print(f"Number of trainable parameters 3 in 'SS' mode: {trainable_params3}")
