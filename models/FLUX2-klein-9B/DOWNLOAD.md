# FLUX.2 Klein Model Weights

## Quick Download (recommended)

```bash
# From the flux-mlx root directory:
huggingface-cli download black-forest-labs/FLUX.2-klein --local-dir models/FLUX2-klein-9B
```

## Manual Download

Download from: https://huggingface.co/black-forest-labs/FLUX.2-klein

Place files in this folder structure:

```
models/FLUX2-klein-9B/
├── transformer/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors    (~7.2 GB)
├── text_encoder/
│   ├── config.json
│   ├── model.safetensors.index.json
│   └── model-00001-of-00002.safetensors (+ 00002)
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors    (~160 MB)
├── tokenizer/
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── scheduler/
│   └── scheduler_config.json
└── model_index.json
```

## License

FLUX.2 Klein is released under Apache 2.0 by Black Forest Labs.
