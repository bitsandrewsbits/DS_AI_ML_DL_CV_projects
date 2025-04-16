from diffusers import DiffusionPipeline
import torch

trained_S_D_model_dir = "trained_S_D_model"

def init_Stable_Diffusion_Pipeline():
    S_D_pipeline = DiffusionPipeline.from_pretrained(
    trained_S_D_model_dir, torch_dtype = torch.float16,  use_safetensors = True
    ).to("cuda")

def get_generated_PIL_image_obj(user_input: str):
    prompt = "Dog on car."
    PIL_image_obj = S_D_pipeline(prompt).images[0]
    return PIL_image_obj

if __name__ == '__main__':
    init_Stable_Diffusion_Pipeline()
    get_generated_PIL_image_obj()
