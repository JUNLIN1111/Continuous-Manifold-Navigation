import argparse
from Model import VisualCausalFlow
import os
from config import Config
import torch
import torch.nn as nn
import torch.optim as optim
from visualize import Visualizer
from data_generation import AppleGripperDataGenerator

def get_args():
    parser = argparse.ArgumentParser(description="ACLF Latent Planning")
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--data_type", type=str, default="visual", choices=["visual", "state"])
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--load", action="store_true", help="Whether to load a pre-trained model")
    return parser.parse_args()
if __name__ == "__main__":
    args = get_args()
    cfg = Config()


    cfg.ModelConfig.latent_dim = args.latent_dim
    cfg.ModelConfig.epochs = args.epochs  
    cfg.ModelConfig.data_type = args.data_type
    if args.data_type != "visual":
        cfg.ModelConfig.encoder_type = "identity"
    cfg.ModelConfig.action_dim = args.action_dim
    if cfg.ModelConfig.data_type == "image" and cfg.ModelConfig.encoder_type == "identity":
        raise ValueError("Identity encoder cannot be used with image data. Please choose a different encoder type.")
    model = VisualCausalFlow(cfg=cfg)
    data_gen = AppleGripperDataGenerator(cfg=cfg)
    vis = Visualizer(model, cfg=cfg) 
    if args.load and os.path.exists("aclf_model.pth"):
        model.load_state_dict(torch.load("aclf_model.pth"))
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        it, inxt, at = data_gen.generate_data()
        crit = nn.BCELoss()
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            idx = torch.randint(0, cfg.ModelConfig.n_samples, (cfg.ModelConfig.batch_size,))
            batch = (it[idx], at[idx], inxt[idx])
            loss = model.compute_loss(batch)
            loss.backward(); optimizer.step() # backpropagation
            if epoch % 200 == 0: print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), "aclf_model.pth")
    
    vis.visualize_all()
    vis.visualize_multi_action_fields()