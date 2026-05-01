import numpy as np
import torch
from config import Config
class AppleGripperDataGenerator:
    """
        z[0]: arm_x, (both in range [-10, 10])
        z[1]: apple_x (both in range [-10, 10])
        a[0]: arm_velocity ([-2, 2])
        a[1]: gripper_action (0 or 1, 0=open, 1=close) 

        Generate synthetic data for training the model. Each sample consists of a current latent state z_t, an action a_t, and the next latent state z_next.
        The dynamics are defined as follows:
        - The first dimension of the latent state (z[0]) is influenced by the action[0] directly.
        - The second dimension of the latent state (z[1]) is influenced by the action only if the first dimension is close to the second dimension (|z[0] - z[1]| < 1.0) and the action[1] is on
    """
    def __init__(self, cfg=Config()):
        self.cfg = cfg
        self.action_dim = self.cfg.ModelConfig.action_dim
        self.latent_dim = self.cfg.ModelConfig.latent_dim
        self.n_samples = self.cfg.ModelConfig.n_samples
        self.data_type = self.cfg.ModelConfig.data_type
        print(f"[Info]: Datagenerator initialized withaction_dim: {self.action_dim}, latent_dim: {self.latent_dim}, n_samples: {self.n_samples}, data_type: {self.data_type}")
    def render_frame(self,z):
        frame = np.zeros((64, 64))
        # Robot arm z[0] is represented as a vertical line, and the apple z[1] is represented as another vertical line. The closer they are, the more likely the apple will be influenced by the arm's action.
        arm_x = int(np.clip((z[0] + 10) * 3.2, 5, 58))
        frame[30:34, arm_x-2:arm_x+2] = 1.0
        apple_x = int(np.clip((z[1] + 10) * 3.2, 5, 58))
        frame[36:40, apple_x-2:apple_x+2] = 0.8
        return frame

    def generate_data(self):
        """
        n_samples: number of samples to generate
        action_dim: dimension of the action space (1 or 2)
        latent_dim: dimension of the latent space (currently fixed to 2 for this dataset)

        return ((batch_size, channel_dim, H, W), (batch_size, action_dim), (batch_size, channel_dim, H, W)) if data_type is "visual"
        return ((batch_size, latent_dim), (batch_size, action_dim), (batch_size, latent_dim)) if data_type is "non_visual"
        """
        # Generate random latent states and actions, then compute the next latent state according to the defined dynamics. Finally, render images if data_type is "visual" or return raw states if data_type is "non_visual".
        z_t = (torch.rand(self.n_samples, 2) * 20) - 10 
        if self.action_dim == 1:
            a_t = (torch.rand(self.n_samples, 1) * 2 - 1) * 2.0
            z_next = z_t.clone()
            z_next[:, 0] += a_t[:, 0] 
            dist = torch.abs(z_t[:, 0] - z_t[:, 1])
            mask = (dist < 1.0).float()
            z_next[:, 1] += a_t[:, 0] * mask 
        elif self.action_dim == 2: 
            a_t = torch.zeros(self.n_samples, 2)
            a_t[:, 0] = (torch.rand(self.n_samples) * 2 - 1) * 2.0
            a_t[:, 1] = torch.randint(0, 2, (self.n_samples,)).float()
            z_next = z_t.clone()
            z_next[:, 0] += a_t[:, 0] # arm_x is always influenced by action[0]
            dist = torch.abs(z_t[:, 0] - z_t[:, 1])
            mask = ((dist < 1.0) & (a_t[:, 1] == 1)).float() # apple_x is influenced by action[0] only if arm_x is close to apple_x and gripper_action is 1 (close)
            z_next[:, 1] += a_t[:, 0] * mask   # apple_x is influenced by action[0] only if arm_x is close to apple_x and gripper_action is 1 (close)
        
        # According to data_type, we can return either visual data (images) or non-visual data (latent states and actions)
        if self.data_type == "visual":
            img_t_list, img_next_list,a_t_list = [], [],[]
            for i in range(self.n_samples):
                img_t_list.append(self.render_frame(z_t[i].numpy()))
                img_next_list.append(self.render_frame(z_next[i].numpy()))
                a_t_list.append(a_t[i].numpy())
            return (torch.from_numpy(np.array(img_t_list)).float().unsqueeze(1),
                    torch.from_numpy(np.array(img_next_list)).float().unsqueeze(1),
                    torch.from_numpy(np.array(a_t_list)).float())
        else:   
            obs_t_list,obs_next_list,a_t_list = [],[],[]
            for i in range(self.n_samples):
                obs_t_list.append(z_t[i].numpy())
                obs_next_list.append(z_next[i].numpy())
                a_t_list.append(a_t[i].numpy())
            return (torch.from_numpy(np.array(obs_t_list)).float(),
                    torch.from_numpy(np.array(obs_next_list)).float(),
                    torch.from_numpy(np.array(a_t_list)).float())
        
