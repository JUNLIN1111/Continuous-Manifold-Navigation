import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data_generation import AppleGripperDataGenerator
from planner import LatentPlanner
from config import Config
class Visualizer:
    def __init__(self, model,cfg=Config()):
        self.model = model
        self.cfg = cfg
    
    def visualize_multi_action_fields(self, res=40):
        """
        Generates a grid of streamplot visualizations with global labels
        and larger individual plots.
        """
        self.model.eval()
        
        # 1. Get actions from Config
        if self.cfg.ModelConfig.action_dim == 1:
            actions = self.cfg.VisualizerConfig.action_space_Visualizer_1d
        else:
            actions = self.cfg.VisualizerConfig.action_space_visualize_2d
        
        # Select actions to display
        display_actions = actions[::max(1, len(actions)//6)] if len(actions) > 6 else actions
        num_plots = len(display_actions)
        
        # 2. Setup Grid: 2 columns wide, rows calculated automatically
        cols = 2
        rows = (num_plots + cols - 1) // cols
        
        # Increase figsize for much larger graphs
        fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows), sharex=True, sharey=True)
        axes = axes.flatten() # Flatten to iterate easily

        # Define Latent Space Grid
        x_range = np.linspace(-5, 10, res)
        y_range = np.linspace(-5, 10, res)
        X, Y = np.meshgrid(x_range, y_range)
        
        z_grid = torch.zeros((res**2, self.cfg.ModelConfig.latent_dim))
        z_grid[:, 0] = torch.from_numpy(X.flatten()).float()
        z_grid[:, 1] = torch.from_numpy(Y.flatten()).float()

        # 3. Plotting loop
        for i in range(len(axes)):
            ax = axes[i]
            if i < num_plots:
                a_tuple = display_actions[i]
                action_tensor = torch.tensor([a_tuple]).float()
                
                with torch.no_grad():
                    inputs = torch.cat([z_grid, action_tensor.repeat(res**2, 1)], dim=-1)
                    dz = self.model.vf_net(inputs).numpy()
                
                DX = dz[:, 0].reshape(res, res)
                DY = dz[:, 1].reshape(res, res)
                mag = np.sqrt(DX**2 + DY**2)
                
                # Streamplot
                ax.streamplot(X, Y, DX, DY, color=mag, cmap='viridis', 
                                     linewidth=1.2, density=1.1, arrowsize=1.3)
                
                # Reference diagonal
                ax.plot([-5, 10], [-5, 10], color='gray', linestyle='--', alpha=0.5)
                ax.set_title(f"Action (a) = {a_tuple}", fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.2)
            else:
                # Hide empty subplots if num_plots is odd
                ax.axis('off')

        # 4. Global Labels (One X and one Y for the whole figure)
        fig.supxlabel("(z0): Hopefully Arm Latent", fontsize=14)
        fig.supylabel("(z1): Hopefully Apple Latent", fontsize=14)
        
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.97]) # Add padding for global labels
        plt.show()

    def visualize_all(self):
        p = LatentPlanner(self.model)
        self.model.eval()
        data_gen = AppleGripperDataGenerator(self.cfg)
        # ---------------------------------------------------------
        # 1. 精简版：起点、终点、边界点（批量处理，无重复代码）
        # ---------------------------------------------------------
        # 定义所有需要渲染的坐标点
        points = {
            "start": [8.0, 2.0],
            "goal": [6.0, 7.0],
            "bound1": [10.0, 10.0],
            "bound2": [-10.0, -10.0],
            "bound3": [10.0, -10.0],
            "bound4": [-10.0, 10.0]
        }
        rendered = {k: data_gen.render_frame(v) for k, v in points.items()}
        if self.cfg.ModelConfig.data_type == "visual":
            s = torch.from_numpy(rendered["start"]).float()
            g = torch.from_numpy(rendered["goal"]).float()
        else:
            s = torch.tensor(points["start"]).float()
            g = torch.tensor(points["goal"]).float()

        # 边界图像统一处理：增加维度 → 浮点型
        def to_tensor(img):
            return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        
        bounds = [to_tensor(rendered[k]) for k in ["bound1", "bound2", "bound3", "bound4"]]

        # A* 规划
        z_path, a_path, z_s, z_g = p.planner_astar(s, g)

        # 编码所有边界点，计算可视化范围
        if self.cfg.ModelConfig.data_type == "visual":
            z_bounds = [self.model.encoder(b).detach().numpy()[0] for b in bounds]
        else:
            z_bounds = [torch.tensor(points[k]).float().numpy() for k in ["bound1", "bound2", "bound3", "bound4"]]  

        column = max(abs(z[0]) for z in z_bounds)
        row = max(abs(z[1]) for z in z_bounds)

        # ---------------------------------------------------------
        # 2. 综合看板：轨迹、向量场、重构检查、规划序列
        # ---------------------------------------------------------
        fig1 = plt.figure(figsize=(18, 9))
        ax1 = fig1.add_subplot(2, 3, 1)
        res = 20

        #  latent 空间网格（自动适配边界，无需手动写范围）
        x, y = np.meshgrid(np.linspace(-column, column, res), np.linspace(-row, row, res))
        z_grid = torch.zeros((res**2, self.cfg.ModelConfig.latent_dim))
        z_grid[:, 0], z_grid[:, 1] = torch.from_numpy(x.flatten()), torch.from_numpy(y.flatten())
        if self.cfg.ModelConfig.action_dim == 1:
            action = torch.tensor([[2.0]])  
        else:
            action = torch.tensor([[2.0,1]])  
        with torch.no_grad():
            dz = self.model.vf_net(torch.cat([z_grid, action.repeat(res**2, 1)], dim=-1))

        # 向量场 + 轨迹绘制
        ax1.quiver(x, y, dz[:, 0], dz[:, 1], color='blue', alpha=0.2)
        p_np = np.array(z_path)
        ax1.plot(p_np[:, 0], p_np[:, 1], 'ro-', markersize=3, label='A* Path')
        ax1.scatter(z_s[0], z_s[1], c='green', s=100, label='Start')
        ax1.scatter(z_g[0], z_g[1], c='red', marker='x', s=100, label='Goal')
        ax1.set_title(f"Latent Trajectory & Vector Field(action={action.squeeze().tolist()})")
        ax1.legend()

        # subplot: Reconstruction Check (GT vs Predicted)
        if self.cfg.ModelConfig.data_type == "visual":
            random_idx = np.random.randint(0, self.cfg.ModelConfig.batch_size)

            img_t, img_next, a_t = data_gen.generate_data()
            img_visual = img_t[random_idx].unsqueeze(0)
            img_visual_next = img_next[random_idx].unsqueeze(0)
            at_visual = a_t[random_idx].unsqueeze(0)
            print(f"[Info] Shape of img_t: {img_visual.shape}, Shape of a_t: {at_visual.shape}")
            with torch.no_grad():
                _, z_next_p = self.model(img_visual, at_visual)
                recon = self.model.decoder(z_next_p)
            ax2 = fig1.add_subplot(2, 3, 2); ax2.imshow(img_visual_next.squeeze(), cmap='gray'); ax2.set_title("GT Next Frame")
            ax3 = fig1.add_subplot(2, 3, 3); ax3.imshow(recon.squeeze(), cmap='gray'); ax3.set_title("Model Prediction")

        
        if len(z_path) > 1  and self.cfg.ModelConfig.data_type == "visual":
            show_num = min(5, len(z_path))
            indices = np.linspace(0, len(z_path)-1, show_num, dtype=int)
            for i, idx in enumerate(indices):
                ax = fig1.add_subplot(2, 5, 5 + i + 1)
                zv = torch.tensor(z_path[idx]).unsqueeze(0).float()
                with torch.no_grad():
                    im = self.model.decoder(zv).squeeze().numpy()
                ax.imshow(im, cmap='gray')
                ax.set_title(f"Plan Step {idx}")
                ax.axis('off')

        # ---------------------------------------------------------
        # 3. 潜空间分布分析 (15x15 基础网格 + 3-4 高密度采样点)
        # ---------------------------------------------------------
        # 1. 基础网格 (15x15)
        steps = 15
        arm_range = np.linspace(-10, 10, steps)
        apple_range = np.linspace(-10, 10, steps)
        latent_grid = np.zeros((steps, steps, 2))
        


        print("[Info] 扫描物理空间映射中...")
        with torch.no_grad():
            # 扫描基础网格
            for i, apple_x in enumerate(apple_range):
                for j, arm_x in enumerate(arm_range):
                    if self.cfg.ModelConfig.data_type == "visual":
                        img = torch.from_numpy(data_gen.render_frame([arm_x, apple_x])).float().unsqueeze(0).unsqueeze(0)
                        latent_grid[i, j] = self.model.encoder(img).squeeze(0).numpy()
                    else:
                        obs = torch.tensor([arm_x, apple_x]).float().unsqueeze(0)
                        latent_grid[i, j] = self.model.encoder(obs).squeeze(0).numpy()

        # 展开基础网格数据
        flat_latent = latent_grid.reshape(-1, 2)
        arm_coords_2d, apple_coords_2d = np.meshgrid(arm_range, apple_range)
        flat_arm_pos = arm_coords_2d.flatten()
        flat_apple_pos = apple_coords_2d.flatten()

        # 绘图：白底、简洁风格
        fig2, (ax_arm, ax_apple) = plt.subplots(1, 2, figsize=(20, 8))
        
        for ax, color_data, title, label_txt in zip(
            [ax_arm, ax_apple], 
            [flat_arm_pos, flat_apple_pos], 
            ["Heatmap: Arm Position", "Heatmap: Apple Position"],
            ["Physical X position (Arm)", "Physical X position (Apple)"]
        ):
            # 1. 绘制 15x15 的灰色参考连线
            for i in range(steps):
                ax.plot(latent_grid[i, :, 0], latent_grid[i, :, 1], color='gray', lw=0.5, alpha=0.2)
                ax.plot(latent_grid[:, i, 0], latent_grid[:, i, 1], color='gray', lw=0.5, alpha=0.2)
            
            # 2. 绘制基础网格点 (带黑边)
            sc = ax.scatter(flat_latent[:, 0], flat_latent[:, 1], 
                            c=color_data, cmap='viridis', s=30,
                            edgecolors='black', linewidths=0.5, alpha=0.7, zorder=2)
            
            
            # 4. 侧边栏与辅助线
            plt.colorbar(sc, ax=ax).set_label(label_txt)
            ax.set_title(title)
            ax.set_xlabel("Latent Dimension 1 ($z_1$)")
            ax.set_ylabel("Latent Dimension 2 ($z_2$)")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_facecolor('white')

        plt.tight_layout()
        plt.show()