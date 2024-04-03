import torch
import torch.nn as nn

# Define the model
mineclip_dim = 512
latent_dim = 256  # experiment with this
hidden_dim = 512  # experiment with this


# Define some helper functions to load the model.
def load_cvae_model(vae_info):
    """Load the VAE model from the given path."""
    # Extract the model parameters.
    visual_goal_dim = vae_info['visual_goal_dim']
    text_dim = vae_info['text_dim']
    current_img_dim = vae_info['current_img_dim']
    goal_img_dim = vae_info['goal_img_dim']
    hidden_dim = vae_info['hidden_dim']
    latent_dim = vae_info['latent_dim']

    model_path = vae_info['model_path']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(visual_goal_dim=visual_goal_dim, text_dim=text_dim, current_img_dim=current_img_dim, goal_img_dim=goal_img_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# Define the model as a simple conditional MLP VAE.
class CVAE(nn.Module):
    def __init__(self, visual_goal_dim=512, text_dim=512, current_img_dim=512, goal_img_dim=512, hidden_dim=256, latent_dim=256):
        super().__init__()
        # Configuration
        self.current_img_dim = current_img_dim
        self.goal_img_dim = goal_img_dim
        self.text_dim = text_dim
        self.visual_goal_dim = visual_goal_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(visual_goal_dim + text_dim + current_img_dim + goal_img_dim, 2 * hidden_dim),
            torch.nn.LayerNorm(2 * hidden_dim),
            torch.nn.ReLU(),

            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),

            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            
            torch.nn.Linear(hidden_dim, 2 * latent_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + text_dim + current_img_dim + goal_img_dim, 2 * hidden_dim),
            torch.nn.LayerNorm(2 * hidden_dim),
            torch.nn.ReLU(),

            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),

            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, visual_goal_dim)
        )

    def encode(self, visual_goal_embeddings, combined_embeddings):
        """Encode the given visual and text embeddings into a latent vector."""
        # Concatenate the visual and text embeddings.
        x = torch.cat([visual_goal_embeddings, combined_embeddings], dim=1)
        # Encode the concatenated embeddings into a latent vector.
        encoded = self.encoder(x)
        mu, logvar = torch.chunk(encoded, 2, dim=1)
        return mu, logvar

    def sample(self, mu, logvar):
        """Sample a latent vector from the given mu and logvar."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent_vector, combined_embeddings):
        """Decode the given latent vector and text embeddings into a visual embedding."""
        # Concatenate the latent vector and text embeddings.
        x = torch.cat([latent_vector, combined_embeddings], dim=1)
        # Decode the concatenated embeddings into a visual embedding.
        return self.decoder(x)


    def forward(self, visual_goal_embeddings, combined_embeddings, deterministic=False):
        mu, logvar = self.encode(visual_goal_embeddings, combined_embeddings)
        
        if deterministic:
            latent_vector = mu  
        else:
            latent_vector = self.sample(mu, logvar)

        pred_visual_embeddings = self.decode(latent_vector, combined_embeddings)

        return pred_visual_embeddings, mu, logvar
    

    def generate(self, combined_embeddings, deterministic=False):
        if deterministic:
            latent_vector = torch.zeros(combined_embeddings.shape[0], self.latent_dim).to(combined_embeddings.device)
        else:
            latent_vector = torch.randn(combined_embeddings.shape[0], self.latent_dim).to(combined_embeddings.device)
        
        generated_visual_goal_embeddings = self.decode(latent_vector, combined_embeddings)

        return generated_visual_goal_embeddings
