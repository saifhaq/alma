**Built with Llama**

To run this example, one must have access to the Llama 3.1 8B model weights from HuggingFace.
To get access, one has to accept the `LLAMA 3.1 COMMUNITY LICENSE AGREEMENT`. 



```bash
export HF_TOKEN=YOUR_TOKEN
python benchmark_with_dataloader.py --conversions EAGER,EXPORT+EAGER --batch-size 10\
--n-samples 5000
```
