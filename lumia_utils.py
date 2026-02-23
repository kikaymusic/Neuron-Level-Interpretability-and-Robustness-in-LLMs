# lumia_utils.py (minimal-change, import-safe)
import os
import numpy as np

model_name = ""


def _pick_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(model_name_aux, model_size):
    # Minimal fix: update the global model_name used in get_activations_and_attention
    global model_name
    model_name = model_name_aux

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import GPTNeoXForCausalLM, BitsAndBytesConfig

    device = _pick_device()

    print(f"getting model (device={device})")

    # Tokenizer always loads the same way
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Case 1: not 12b
    if "12b" not in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Move to device (cuda/mps/cpu)
        model = model.to(device)
        return model, tokenizer

    # Case 2: 12b
    # Only try bitsandbytes quantization if:
    # - bitsandbytes is installed
    # - and we are on CUDA (Mac MPS/CPU won't use it)
    try:
        import bitsandbytes  # noqa: F401
        has_bnb = True
    except ImportError:
        has_bnb = False

    if has_bnb and device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )
        model = GPTNeoXForCausalLM.from_pretrained(
            model_name,
            device_map="cuda:0",
            quantization_config=quantization_config,
        )
    else:
        # Fallback for Mac: load normally
        # (If it's too big for RAM, you'll need a smaller model anyway.)
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        model = GPTNeoXForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)

    return model, tokenizer


def get_activations_and_attention(
    model, tokenizer, text, start_layer, end_layer, token, max_token_division
):
    import torch

    tokenizer.pad_token = tokenizer.eos_token

    # Uses the global model_name, same as your original intent
    if "12b" in model_name:
        inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=1024)
    else:
        inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=2048)

    with torch.no_grad():
        try:
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            outputs = model.forward(inputs, output_hidden_states=True)

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA out of memory. Printing memory status and attempting to clear cache.")
                for i in range(torch.cuda.device_count()):
                    print(f"Memory summary for GPU {i}:")
                    print(torch.cuda.memory_summary(device=i))
                torch.cuda.empty_cache()
            raise e

        last_tokens = []
        to_plot_activations = []

        for i in range(start_layer, end_layer):
            numpy_arr = outputs["hidden_states"][i][:, token].detach().cpu().numpy()
            mean = outputs["hidden_states"][i].mean(axis=1)

            last_tokens.append(mean.cpu())

        last_token_activations = torch.stack(last_tokens)

    return last_token_activations


def get_attention_weights(model, tokenizer, text, start_layer, end_layer):
    import torch

    inputs = tokenizer.encode(text, return_tensors="pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)

    try:
        # Run model forward pass with attention weights enabled
        outputs = model(inputs, output_attentions=True)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory. Printing memory status and attempting to clear cache.")
            for i in range(torch.cuda.device_count()):
                print(f"Memory summary for GPU {i}:")
                print(torch.cuda.memory_summary(device=i))
            print(torch.cuda.memory_summary())
            torch.cuda.empty_cache()
        raise e

    attention_weights = []
    for i in range(start_layer, end_layer):
        # Extract attention weights for the specified layer
        if i < len(outputs.attentions):
            layer_attention = outputs.attentions[i].detach().cpu()
            print(layer_attention)
            attention_weights.append(layer_attention)
        else:
            print(f"Layer {i} is out of range. Skipping.")

    return attention_weights


class Classifier:
    # Minimal-change: keep same structure but lazy import TF inside methods
    def __init__(self, input_dim):
        import tensorflow as tf
        from tensorflow.keras import layers, initializers

        super(Classifier, self).__init__()
        self.tf = tf
        self.layers = layers

        self.fc1 = layers.Dense(
            768, kernel_initializer=initializers.HeUniform(), input_dim=input_dim
        )
        self.fc2 = layers.Dense(256, kernel_initializer=initializers.HeUniform())
        self.fc3 = layers.Dense(1)
        self.dropout = layers.Dropout(0.5)

    def call(self, x, training=False):
        tf = self.tf
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        x = self.fc3(x)
        x = tf.sigmoid(x)
        return x


def build_classifier(input_dim):
    import tensorflow as tf
    from tensorflow.keras import optimizers, losses

    classifier = Classifier(input_dim)

    # Define the loss function
    criterion = losses.BinaryCrossentropy(from_logits=False)

    # Define the optimizer
    optimizer = optimizers.Adam()

    # Compile the model
    classifier.compile(optimizer=optimizer, loss=criterion, metrics=["accuracy"])

    return classifier