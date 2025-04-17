from diffusers import DiffusionPipeline
import torch

trained_S_D_model_dir = "trained_S_D_model"

def get_Stable_Diffusion_Pipeline():
    pipeline = DiffusionPipeline.from_pretrained(
        trained_S_D_model_dir, torch_dtype = torch.float16,
        use_safetensors = True
    ).to("cuda")
    return pipeline

def get_generated_PIL_image_obj(pipeline, user_input: str):
    PIL_image_obj = pipeline(user_input).images[0]
    return PIL_image_obj

if __name__ == '__main__':
    S_D_pipeline = get_Stable_Diffusion_Pipeline()
    user_input = "Cat reads a book."
    result_image_obj = get_generated_PIL_image_obj(S_D_pipeline, user_input)
