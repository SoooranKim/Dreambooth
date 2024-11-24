import torch
from diffusers import StableDiffusionPipeline

# 모델 경로 설정
model_path = "/root/Dreambooth/output"

try:
    # 파이프라인 로드
    pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")  # GPU 사용

    # 인퍼런스 프롬프트
    prompt = "A photo of a [fluffy frog] plush sitting in a library surrounded by books"
    image = pipeline(prompt, num_inference_steps=100, guidance_scale=8).images[0]

    # 결과 저장
    output_path = "/root/Dreambooth/output/zfrog30.png"
    image.save(output_path)


finally:
    # 메모리 초기화
    del pipeline
    torch.cuda.empty_cache()
