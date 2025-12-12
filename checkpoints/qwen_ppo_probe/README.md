---
base_model: /scratch/victord2/COMP545/project/models/qwen2.5-3b-instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:/scratch/victord2/COMP545/project/models/qwen2.5-3b-instruct
- lora
- transformers
---
### Framework versions

- PEFT 0.18.0