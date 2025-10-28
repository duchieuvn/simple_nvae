import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from nvae.utils import add_sn
from nvae.vae_celeba import NVAE


def generate_new_image(model, device):
    """Generates a single image from a random latent vector."""
    with torch.no_grad():
        z = torch.randn((1, 512, 2, 2)).to(device)
        gen_img, _ = model.decoder(z)
        gen_img = gen_img.permute(0, 2, 3, 1)
        gen_img = gen_img[0].cpu().numpy() * 255
        gen_img = gen_img.astype(np.uint8)
    return gen_img


if __name__ == '__main__':
    # --- Model Setup ---
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = NVAE(z_dim=512, img_dim=(64, 64))
    model.apply(add_sn)
    model.to(device)

    # Note: You might want to update this checkpoint path
    checkpoint_path = "checkpoints/ae_ckpt_90_0.809145.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.eval()

    # --- Matplotlib Figure Setup ---
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.axis('off')

    # Display the first image
    im = ax.imshow(generate_new_image(model, device))

    # --- Button Callback ---
    def on_click(event):
        new_img = generate_new_image(model, device)
        im.set_data(new_img)
        plt.draw()

    # --- Button Widget ---
    ax_button = plt.axes([0.35, 0.05, 0.3, 0.075])
    button = Button(ax_button, 'Generate New Image')
    button.on_clicked(on_click)

    plt.show()