from diffusers import DiffusionPipeline
import torch

finetuned_S_D_model_dir = "finetuned_S_D_LoRA_model"

def get_Stable_Diffusion_Pipeline(LoRA_weights_dir: str):
    pipeline = DiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker = None
    )
    pipeline.to("cuda")
    pipeline.load_lora_weights(LoRA_weights_dir)
    return pipeline

def get_generated_PIL_image_obj(pipeline, user_input: str):
    PIL_image_obj = pipeline(user_input).images[0]
    return PIL_image_obj

if __name__ == '__main__':
    S_D_pipeline = get_Stable_Diffusion_Pipeline(finetuned_S_D_model_dir)
    user_input = "Cat reads a book."
    result_image_obj = get_generated_PIL_image_obj(S_D_pipeline, user_input)
