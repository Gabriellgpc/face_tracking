import torchvision
import torch

def main():
    base_model = torchvision.models.mobilenet_v3_large(pretrained=True)

    backbone = torch.nn.Sequential(
        *list(base_model.children())[:-1],
        torch.nn.Flatten(),
        )

    dummy_input = torch.randn(1, 3, 256, 256)

    torch.onnx.export(backbone,
                      dummy_input,
                      "/workspace/backbone-mobilenetv3-large.onnx",
                      verbose=True,
                      input_names=['input_image'],
                      output_names=['output_encoder']
                      )

    # convert to onnx to openvino ir
    # mo --input_model /workspace/backbone-mobilenetv3-large.onnx --compress_to_fp16

if __name__ == '__main__':
    main()