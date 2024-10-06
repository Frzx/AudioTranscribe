import torch
import transformers
from transformers.modeling_utils import no_init_weights
from transformers import AutoConfig
import safetensors
from pathlib import Path

from shuka_model.shuka_model import ShukaModel

class AudioTranscriber:
    shuka_config_auto = AutoConfig.from_pretrained("sarvamai/shuka_v1", trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, encoder_safetensor_path, whisper_model_safetensor_):
 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load configuration for the Shuka model
        self.shuka_config_auto = AutoConfig.from_pretrained("sarvamai/shuka_v1", trust_remote_code=True)

        # Load encoder, decoder and decoder tokenizer
        self.encoder = self.get_encoder(encoder_safetensor_path)
        self.decoder = self.get_decoder(whisper_model_safetensor_)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.decoder.config._name_or_path)

    @classmethod
    def get_encoder(cls, safetensors_path):
        # Load shuka model
        with no_init_weights():
            shuka_model = ShukaModel(cls.shuka_config_auto)

        with torch.no_grad():
            # Get shuka model audio tower layers
            audio_tower = shuka_model.audio_tower.to(cls.device)

            # Load safe tensors
            model_weights_001 = safetensors.torch.load_file(safetensors_path)

            # Move loaded tensors to GPU
            audio_tower_weights = {
                layer_name.replace("audio_tower.", ""): layer_weights.to(cls.device)
                for layer_name, layer_weights in model_weights_001.items()
                if layer_name.startswith("audio_tower")
            }

            # Load the weights in audio_tower layers
            msg = audio_tower.load_state_dict(audio_tower_weights)
            print("Encoder Safetensor load",msg)

            # Freeing up memory
            del audio_tower_weights
            del model_weights_001
            torch.cuda.empty_cache()

            return audio_tower

    @classmethod
    def get_decoder(cls, safetensors_path):
        # Load whisper model
        with transformers.modeling_utils.no_init_weights():
            whisper_model = transformers.AutoModel.from_pretrained("seba3y/whisper-large-v2-16fp")

        # Get decoder
        decoder = whisper_model.decoder.to(cls.device)

        # Load safe tensors
        model_weights = safetensors.torch.load_file(safetensors_path)

        # Move and convert tensors to GPU and BF16
        decoder_layers_weights = {
            layer_name.replace("model.decoder.", ""): layer_tensors.to(cls.device, dtype=torch.bfloat16)
            for layer_name, layer_tensors in model_weights.items()
            if layer_name.startswith("model.decoder")
        }

        # Load the weights in the decoder layers
        msg = decoder.load_state_dict(decoder_layers_weights)
        print("Decoder saftensor load",msg)

        # Freeing up memory
        del decoder_layers_weights
        del model_weights
        torch.cuda.empty_cache()

        return decoder

    def __call__(self, processed_audio_input):
        # Encoder forward
        encoder_output = self.encoder.forward(processed_audio_input.to(self.device))

        # Decoder forward
        input_ids = self.tokenizer("<|startoftranscript|>", return_tensors="pt").input_ids.to(self.device)
        max_length = 100
        generated_tokens = []

        for step in range(max_length):
            # Run the decoder
            with torch.no_grad():
                outputs = self.decoder(
                    input_ids=input_ids,
                    encoder_hidden_states=encoder_output.last_hidden_state,
                )

                last_hidden_state = outputs.last_hidden_state
                logits = last_hidden_state[:, -1, :]
                # Project to vocab size using the decoder's embed_tokens layer
                logits = logits @ self.decoder.embed_tokens.weight.T
                last_token_logits = logits
                predicted_token_id = torch.argmax(last_token_logits, dim=-1).unsqueeze(1)

                decoded_token = self.tokenizer.decode(predicted_token_id.item())
                generated_tokens.append(decoded_token)

                # Check for end-of-sequence token
                if predicted_token_id.item() == self.tokenizer.eos_token_id:
                    break

                # Update input_ids for the next step
                input_ids = torch.cat([input_ids, predicted_token_id], dim=1)

        return ''.join(generated_tokens)

